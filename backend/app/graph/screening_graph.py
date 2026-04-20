"""
Resume Screening Analysis Pipeline using LangGraph and LLM

This module implements a multi-stage resume screening workflow that:
1. Structures both job descriptions and resumes using LLM-powered extraction
2. Performs semantic matching using vector embeddings (Chroma)
3. Conducts gap analysis to identify skill mismatches
4. Generates personalized outreach emails based on match quality

Architecture:
- ResumeState: Tracks the analysis state for a single resume
- ParentState: Manages batch processing of multiple resumes
- SINGLE_RESUME_GRAPH: Processes one resume through the full pipeline
- PARENT_GRAPH: Routes multiple resumes to the single-resume graph using Send

Key Technologies:
- LangGraph: State machine orchestration for complex LLM workflows
- OpenAI: GPT-4o-mini for structured data extraction and analysis
- Chroma: Vector database for semantic similarity search
- LangChain: Document processing and text utilities
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict, Annotated
import operator

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.core.settings import get_settings, load_config
from app.models.schemas import CandidateCard, EmailDraft, MatchAnalysis, ResumeAnalysisState, StructuredProfile
from app.services.scoring import build_jd_query, role_title, weighted_match_percentage
from app.utils.text import clean_text, extract_email, maybe_extract_name_from_header, normalize_list, split_text_for_docs



class ResumeState(TypedDict):
    """
    State container for a single resume analysis workflow.
    
    This TypedDict tracks all data flowing through the single-resume graph nodes:
    - Input data: jd_text, resume_text, temperature, file_name
    - Structured outputs: structured_jd, structured_resume (LLM-extracted data)
    - Vector search results: retrieved_context (relevant resume chunks)
    - Analysis results: match_analysis, match_percentage, strengths, gaps, etc.
    - Email generation outputs: email_subject, email_body
    - Metadata: candidate_name, candidate_email, job_title
    """
    jd_text: str
    resume_text: str
    temperature: float
    file_name: str | None
    structured_jd: dict[str, Any]
    structured_resume: dict[str, Any]
    retrieved_context: str
    match_analysis: dict[str, Any]
    match_percentage: float
    strengths: list[str]
    gaps: list[str]
    missing_but_relevant: list[str]
    soft_skill_alignment: list[str]
    email_draft: dict[str, Any]
    email_subject: str
    email_body: str
    candidate_name: str | None
    candidate_email: str | None
    processed: bool
    email_pending: bool
    job_title: str | None
    match_breakdown: dict[str, float]


class ParentState(TypedDict):
    """
    State container for batch processing multiple resumes.
    
    Uses Annotated reducers for aggregation:
    - candidate_cards: accumulated list using operator.add
    - processed/email_pending: OR logic to track if ANY resume needs attention
    """
    jd_text: str
    temperature: float
    resume_items: list[dict[str, Any]]
    candidate_cards: Annotated[list[dict[str, Any]], operator.add]
    processed: Annotated[bool, lambda old, new: bool(old) or bool(new)]
    email_pending: Annotated[bool, lambda old, new: bool(old) or bool(new)]


def _llm(temperature: float = 0.0) -> ChatOpenAI:
    """
    Initialize an OpenAI LLM client with the specified temperature.
    
    Temperature controls output randomness:
    - 0.0: Deterministic, consistent extraction (used for structured data extraction)
    - 0.4: Balanced creativity for email generation
    - Higher values: More creative/varied responses
    """
    settings = get_settings()
    return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=temperature)


def _structure_text(source_type: Literal["job_description", "resume"], text: str):
    """
    Extract and structure recruiting data from job descriptions or resumes using LLM.
    
    This node uses GPT-4o-mini with structured output mode to parse unstructured text
    into a standardized StructuredProfile schema. This enables:
    - Consistent data format for downstream analysis
    - Extraction of: skills, experience, seniority, soft skills, requirements
    - Fallback extraction for email and name if LLM extraction fails
    
    For resumes, applies secondary extraction logic:
    - If LLM didn't extract email, use regex-based extraction from raw text
    - If LLM didn't extract name, parse header section for candidate name
    """
    llm = _llm(0.0)
    parser = llm.with_structured_output(StructuredProfile)
    label = "job description" if source_type == "job_description" else "resume"
    prompt = f"""
You are extracting structured recruiting data from a {label}.

Return JSON with:
- source_type: {source_type}
- title: role title or candidate role focus
- summary: concise 2-4 sentence summary
- skills: key technical skills, frameworks, tools
- years_experience: estimated years of relevant experience if visible, else null
- seniority_level: junior/mid/senior/staff/lead/manager or null
- soft_skills: collaboration, ownership, communication, leadership, etc.
- must_have: hard requirements if present
- nice_to_have: optional skills if present
- candidate_name: only for resume, else null
- candidate_email: only for resume, else null
- evidence: short snippets that justify the extraction

Text:
{text}
"""
    result = parser.invoke(prompt)
    if source_type == "resume":
        if not result.candidate_email:
            result.candidate_email = extract_email(text)
        if not result.candidate_name:
            result.candidate_name = maybe_extract_name_from_header(text)
    return result



def extract_and_clean(state: ResumeState) -> ResumeState:
    """
    Node 1: Extract and Clean
    
    Purpose: Parse raw job description and resume text into structured data
    
    Process:
    1. Clean up whitespace, special characters, and formatting inconsistencies
    2. Use LLM to extract structured information from job description
    3. Use LLM to extract structured information from resume
    4. Fallback to regex-based extraction if LLM misses email/name
    5. Extract candidate metadata (name, email, job title)
    
    Output: Structured profiles ready for semantic analysis
    """
    jd_text = clean_text(state["jd_text"])
    resume_text = clean_text(state["resume_text"])
    structured_jd = _structure_text("job_description", jd_text)
    structured_resume = _structure_text("resume", resume_text)
    return {
        **state,
        "jd_text": jd_text,
        "resume_text": resume_text,
        "structured_jd": structured_jd.model_dump(),
        "structured_resume": structured_resume.model_dump(),
        "candidate_name": structured_resume.candidate_name or maybe_extract_name_from_header(resume_text),
        "candidate_email": structured_resume.candidate_email or extract_email(resume_text),
        "job_title": structured_jd.title or structured_jd.summary,
    }


def semantic_match(state: ResumeState) -> ResumeState:
    """
    Node 2: Semantic Matching
    
    Purpose: Compare resume content semantically against job requirements using embeddings
    
    Process:
    1. Build a comprehensive query from all job description requirements
    2. Split resume into chunks and generate embeddings using OpenAI text-embedding-3-small
    3. Store embeddings in Chroma vector database for similarity search
    4. Retrieve top 5 resume chunks most relevant to job requirements
    5. Use LLM to score technical skills, experience, and seniority alignment (0-100 scale)
    6. Calculate weighted match percentage based on scoring weights from config
    
    Key Feature: Uses semantic understanding, not keyword matching - recognizes synonyms,
    related technologies, and transferable skills
    
    Output: Match analysis with individual scores and overall match percentage
    """
    jd = StructuredProfile.model_validate(state["structured_jd"])
    resume = StructuredProfile.model_validate(state["structured_resume"])
    jd_query = build_jd_query(jd)

    settings = get_settings()
    chroma_root = Path(settings.chroma_dir)
    chroma_root.mkdir(parents=True, exist_ok=True)

    resume_docs = split_text_for_docs(state["resume_text"])
    for idx, doc in enumerate(resume_docs):
        doc.metadata = {**doc.metadata, "chunk": idx, "file_name": state.get("file_name")}

    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key, model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=resume_docs,
        embedding=embeddings,
        persist_directory=str(chroma_root / f"resume_{abs(hash(state['resume_text'])) % 10_000_000}"),
    )

    retrieved = vectorstore.similarity_search(jd_query, k=5)
    retrieved_context = "\n\n".join(f"Chunk {idx + 1}: {doc.page_content[:1200]}" for idx, doc in enumerate(retrieved))

    llm = _llm(0.0)
    parser = llm.with_structured_output(MatchAnalysis)
    prompt = f"""
You are comparing a candidate resume to a job description using semantic reasoning, not keyword counting.

Job description structured JSON:
{jd.model_dump_json(indent=2)}

Resume structured JSON:
{resume.model_dump_json(indent=2)}

Retrieved resume evidence:
{retrieved_context}

Task:
1. Score technical_skills_score from 0-100 based on semantic fit of core competencies.
2. Score experience_years_score from 0-100 based on whether the candidate's experience matches the JD requirement.
3. Score seniority_alignment_score from 0-100 based on seniority fit.
4. Provide strengths, gaps, missing_but_relevant, and soft_skill_alignment.
5. Keep the scores honest; do not inflate.
"""
    analysis = parser.invoke(prompt)
    match_percentage, breakdown = weighted_match_percentage(analysis)
    return {
        **state,
        "retrieved_context": retrieved_context,
        "match_analysis": analysis.model_dump(),
        "match_percentage": match_percentage,
        "strengths": normalize_list(analysis.strengths)[:5],
        "gaps": normalize_list(analysis.gaps)[:5],
        "missing_but_relevant": normalize_list(analysis.missing_but_relevant)[:5],
        "soft_skill_alignment": normalize_list(analysis.soft_skill_alignment)[:5],
        "match_breakdown": breakdown,
    }


def gap_analysis(state: ResumeState) -> ResumeState:
    """
    Node 3: Gap Analysis
    
    Purpose: Refine and contextualize the match analysis findings
    
    Process:
    1. Review the initial match analysis alongside structured data
    2. Use LLM to provide refined insights on:
       - Key strengths the candidate brings
       - Critical skill gaps or missing experience
       - Relevant but missing skills (things that would be nice to have)
       - Soft skill alignment inferred from resume
    3. Preserve numeric scores from semantic_match (no re-scoring)
    4. Fallback to previous analysis if LLM provides empty results
    
    This step adds human-like reasoning and context to the raw scoring
    """
    jd = StructuredProfile.model_validate(state["structured_jd"])
    resume = StructuredProfile.model_validate(state["structured_resume"])
    llm = _llm(0.0)
    parser = llm.with_structured_output(MatchAnalysis)
    prompt = f"""
You are doing gap analysis between a job description and a resume.

Job description:
{jd.model_dump_json(indent=2)}

Resume:
{resume.model_dump_json(indent=2)}

Current match analysis:
{state['match_analysis']}

Return refined JSON with:
- strengths: three to six strengths
- gaps: two to five gaps or missing skills
- missing_but_relevant: skills the candidate lacks but the role likely values
- soft_skill_alignment: soft skills that appear aligned or inferred from evidence
- notes: one concise hiring note
Do not change the numeric scores; preserve them.
"""
    refined = parser.invoke(prompt)
    return {
        **state,
        "strengths": normalize_list(refined.strengths)[:5] or state["strengths"],
        "gaps": normalize_list(refined.gaps)[:5] or state["gaps"],
        "missing_but_relevant": normalize_list(refined.missing_but_relevant)[:5] or state["missing_but_relevant"],
        "soft_skill_alignment": normalize_list(refined.soft_skill_alignment)[:5] or state["soft_skill_alignment"],
    }


def email_generation(state: ResumeState) -> ResumeState:
    """
    Node 4: Email Generation
    
    Purpose: Generate personalized outreach emails for HR to send to candidates
    
    Process:
    1. Load email templates and sender configuration
    2. Extract job title for personalization
    3. Use LLM with configurable temperature to generate email tone:
       - Low temp (0.0): Formal, professional tone for corporate/government
       - High temp (1.0): Warm, enthusiastic tone for startups/creative roles
    4. Create subject line from template with job title substitution
    5. Draft personalized body incorporating:
       - Candidate strengths
       - Match percentage context
       - Role-specific personalization
       - Call-to-action
    6. Fallback to template if LLM generation fails
    
    Output: Email subject and body ready for HR to review and send
    """
    config = load_config()
    jd = StructuredProfile.model_validate(state["structured_jd"])
    job_title = role_title(jd)
    temp = float(state.get("temperature", 0.4))

    llm = _llm(temp)
    parser = llm.with_structured_output(EmailDraft)
    subject = config["email_templates"]["subject_template"].replace("[Job_Title]", job_title)
    prompt = f"""
Write a personalized outreach email for a recruiter co-pilot application.

Sender: {config['sender_email']}
Candidate name: {state.get('candidate_name') or 'Candidate'}
Candidate email: {state.get('candidate_email') or ''}
Job title: {job_title}
Match percentage: {state['match_percentage']}
Strengths: {state['strengths']}
Gaps: {state['gaps']}
Missing but relevant: {state['missing_but_relevant']}
Soft skill alignment: {state['soft_skill_alignment']}

Tone instructions:
- Temperature near 0.0 = highly professional/formal, conservative, banking/gov style.
- Temperature near 1.0 = enthusiastic, warm, startup/creative style.
- Keep the email concise, personalized, and human.
- Do not mention raw scoring mechanics.

Use the provided subject line exactly:
{subject}
"""
    drafted = parser.invoke(prompt)
    body = drafted.body.strip() or config["email_templates"]["body_template"].replace("[Name]", state.get("candidate_name") or "Candidate")
    return {
        **state,
        "email_subject": subject,
        "email_body": body,
        "processed": True,
        "email_pending": True,
    }


def finalize_resume(state: ResumeState) -> dict[str, Any]:
    """
    Node 5: Finalize Resume (Terminal Node)
    
    Purpose: Compile all analysis results into a final CandidateCard output
    
    Process:
    1. Validate all required fields are present
    2. Create structured CandidateCard with:
       - Candidate contact information
       - Match score and breakdown
       - Key findings (strengths, gaps, soft skills)
       - Generated email for outreach
       - Both structured profiles for reference
    3. Return card as serializable dict
    
    This is the final output that gets sent to the frontend
    """
    jd = StructuredProfile.model_validate(state["structured_jd"])
    card = CandidateCard(
        candidate_name=state.get("candidate_name"),
        candidate_email=state.get("candidate_email"),
        job_title=role_title(jd),
        match_percentage=float(state["match_percentage"]),
        strengths=state.get("strengths", [])[:3],
        gaps=state.get("gaps", [])[:2],
        missing_but_relevant=state.get("missing_but_relevant", [])[:5],
        soft_skill_alignment=state.get("soft_skill_alignment", [])[:5],
        email_subject=state.get("email_subject", ""),
        email_body=state.get("email_body", ""),
        processed=True,
        email_pending=True,
        structured_jd=jd,
        structured_resume=StructuredProfile.model_validate(state["structured_resume"]),
        match_breakdown=state.get("match_breakdown", {}),
    )
    return {"candidate_card": card.model_dump(), "processed": True, "email_pending": True}



def build_single_resume_graph():
    """
    Construct the single-resume analysis graph.
    
    Uses LangGraph's StateGraph to define a DAG (Directed Acyclic Graph) workflow:
    START → extract_and_clean → semantic_match → gap_analysis → email_generation → finalize_resume → END
    
    This deterministic pipeline ensures consistent, reproducible analysis for each resume.
    All state is passed through the graph, enabling easy debugging and state inspection.
    """
    builder = StateGraph(ResumeState)
    builder.add_node("extract_and_clean", extract_and_clean)
    builder.add_node("semantic_match", semantic_match)
    builder.add_node("gap_analysis", gap_analysis)
    builder.add_node("email_generation", email_generation)
    builder.add_node("finalize_resume", finalize_resume)

    builder.add_edge(START, "extract_and_clean")
    builder.add_edge("extract_and_clean", "semantic_match")
    builder.add_edge("semantic_match", "gap_analysis")
    builder.add_edge("gap_analysis", "email_generation")
    builder.add_edge("email_generation", "finalize_resume")
    builder.add_edge("finalize_resume", END)
    return builder.compile()


SINGLE_RESUME_GRAPH = build_single_resume_graph()


def analyze_resume_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Process a single resume through the complete analysis pipeline.
    
    This node:
    1. Initializes ResumeState with input data
    2. Invokes the SINGLE_RESUME_GRAPH with the state
    3. Maps graph outputs to ParentState format
    4. Returns aggregated results for batch processing
    """
    resume_state: ResumeState = {
        "jd_text": state["jd_text"],
        "resume_text": state["resume_text"],
        "temperature": float(state.get("temperature", 0.4)),
        "file_name": state.get("file_name"),
        "structured_jd": {},
        "structured_resume": {},
        "retrieved_context": "",
        "match_analysis": {},
        "match_percentage": 0.0,
        "strengths": [],
        "gaps": [],
        "missing_but_relevant": [],
        "soft_skill_alignment": [],
        "email_draft": {},
        "email_subject": "",
        "email_body": "",
        "candidate_name": None,
        "candidate_email": None,
        "processed": False,
        "email_pending": False,
        "job_title": None,
        "match_breakdown": {},
    }
    print("🚀 Starting graph execution")
    result = SINGLE_RESUME_GRAPH.invoke(resume_state)
    print("✅ Graph finished", result)
    candidate_card = {
        "candidate_name": result.get("candidate_name"),   
        "candidate_email": result.get("candidate_email"),
        "job_title": result.get("job_title"),
        "match_percentage": result.get("match_percentage"),
        "strengths": result.get("strengths", []),
        "gaps": result.get("gaps", []),
        "missing_but_relevant": result.get("missing_but_relevant", []),
        "soft_skill_alignment": result.get("soft_skill_alignment", []),
        "email_subject": result.get("email_subject", ""),
        "email_body": result.get("email_body", ""),
        "processed": bool(result.get("processed", False)),
        "email_pending": bool(result.get("email_pending", False)),
        "match_breakdown": result.get("match_breakdown", {}),
    }
    return {
        "candidate_cards": [candidate_card],
        "processed": bool(result.get("processed", False)),
        "email_pending": bool(result.get("email_pending", False)),
    }


def route_resumes(state: ParentState):
    """
    Conditional router that maps each resume item to an analyze_resume task.
    
    Uses LangGraph's Send() to create one task per resume, enabling parallel processing.
    Each task receives the job description and individual resume for analysis.
    """
    return [
        Send(
            "analyze_resume",
            {
                "jd_text": state["jd_text"],
                "resume_text": item["resume_text"],
                "temperature": state.get("temperature", 0.4),
                "file_name": item.get("file_name"),
            },
        )
        for item in state.get("resume_items", [])
    ]


def build_parent_graph():
    """
    Construct the batch processing graph for multiple resumes.
    
    Architecture:
    START → route_resumes → [analyze_resume (parallel tasks)] → END
    
    The parent graph uses conditional routing to create one analyze_resume node
    invocation per resume. LangGraph automatically handles:
    - Parallel execution across CPUs
    - State aggregation using Annotated reducers (list concatenation, OR logic)
    """
    builder = StateGraph(ParentState)
    builder.add_node("analyze_resume", analyze_resume_node)
    builder.add_conditional_edges(START, route_resumes)
    builder.add_edge("analyze_resume", END)
    return builder.compile()


PARENT_GRAPH = build_parent_graph()


def run_analysis(jd_text: str, resume_texts: list[dict[str, Any]], temperature: float = 0.4) -> dict[str, Any]:
    """
    Entry point for the resume screening analysis.
    
    This function orchestrates the entire batch processing pipeline:
    1. Initializes ParentState with job description and resume list
    2. Invokes PARENT_GRAPH which routes each resume through analysis
    3. Returns aggregated results across all resumes
    
    Args:
        jd_text: Job description text to match against
        resume_texts: List of {resume_text, file_name} dicts
        temperature: LLM temperature for email tone (0.0-1.0)
    
    Returns:
        Dictionary with candidate_cards list and processing status
    """
    state: ParentState = {
        "jd_text": jd_text,
        "temperature": temperature,
        "resume_items": resume_texts,
        "candidate_cards": [],
        "processed": False,
        "email_pending": False,
    }
    return PARENT_GRAPH.invoke(state)
