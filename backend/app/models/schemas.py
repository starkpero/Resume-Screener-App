"""Pydantic data models for API requests, responses, and internal processing.

These schemas define the contract between frontend/backend and enforce
data validation using Pydantic's BaseModel.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal
import operator

from pydantic import BaseModel, Field


class ResumeItem(BaseModel):
    """Single resume for batch processing."""
    file_name: str | None = None
    resume_text: str


class AnalysisRequest(BaseModel):
    """User request to analyze resume against job description."""
    jd_text: str = Field(..., min_length=5)
    resume_text: str | None = None
    temperature: float = Field(default=0.4, ge=0.0, le=1.0)


class SendEmailRequest(BaseModel):
    """Request to send candidate outreach email."""
    candidate_email: str
    candidate_name: str | None = None
    job_title: str = "the role"
    subject: str | None = None
    body: str | None = None


class StructuredProfile(BaseModel):
    """
    Standardized profile extracted from either resume or job description.
    
    This is the output of the LLM extraction step. All fields are optional
    to handle cases where information isn't available in the source text.
    """
    source_type: Literal["job_description", "resume"]
    title: str | None = None
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    years_experience: float | None = None
    seniority_level: str | None = None
    soft_skills: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    candidate_name: str | None = None
    candidate_email: str | None = None
    evidence: list[str] = Field(default_factory=list)


class MatchAnalysis(BaseModel):
    """
    Detailed matching analysis between resume and job description.
    
    Three independent scoring dimensions (0-100):
    - technical_skills_score: Core competency alignment
    - experience_years_score: Years of experience match
    - seniority_alignment_score: Career level fit
    
    These scores are combined via weighted_match_percentage()
    """
    technical_skills_score: float = Field(..., ge=0.0, le=100.0)
    experience_years_score: float = Field(..., ge=0.0, le=100.0)
    seniority_alignment_score: float = Field(..., ge=0.0, le=100.0)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    missing_but_relevant: list[str] = Field(default_factory=list)
    soft_skill_alignment: list[str] = Field(default_factory=list)
    notes: str | None = None


class EmailDraft(BaseModel):
    """Generated email draft for candidate outreach."""
    subject: str
    body: str


class CandidateCard(BaseModel):
    """
    Final output card for a single analyzed resume.
    
    Contains all analysis results ready for frontend display and email sending.
    Includes both the match scores and supporting context (strengths, gaps, etc.)
    """
    candidate_name: str | None = None
    candidate_email: str | None = None
    job_title: str | None = None
    match_percentage: float
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    missing_but_relevant: list[str] = Field(default_factory=list)
    soft_skill_alignment: list[str] = Field(default_factory=list)
    email_subject: str
    email_body: str
    processed: bool = True
    email_pending: bool = True
    structured_jd: StructuredProfile | None = None
    structured_resume: StructuredProfile | None = None
    match_breakdown: dict[str, float] = Field(default_factory=dict)


class CandidateResults(BaseModel):
    """
    API response containing batch analysis results.
    
    Returns aggregated results across all resumes analyzed in a single request.
    """
    candidate_cards: list[CandidateCard] = Field(default_factory=list)
    processed: bool = False
    email_pending: bool = False
    request_count_remaining: int | None = None
    message: str | None = None


class ResumeAnalysisState(BaseModel):
    """
    Alternative state representation for single resume analysis.
    Uses model-based state instead of TypedDict.
    """
    jd_text: str
    resume_text: str
    temperature: float = 0.4
    file_name: str | None = None
    structured_jd: StructuredProfile | None = None
    structured_resume: StructuredProfile | None = None
    retrieved_context: str = ""
    match_analysis: MatchAnalysis | None = None
    match_percentage: float = 0.0
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    missing_but_relevant: list[str] = Field(default_factory=list)
    soft_skill_alignment: list[str] = Field(default_factory=list)
    email_draft: EmailDraft | None = None
    candidate_name: str | None = None
    candidate_email: str | None = None
    processed: bool = False
    email_pending: bool = False
    job_title: str | None = None
    match_breakdown: dict[str, float] = Field(default_factory=dict)


class ParentGraphState(BaseModel):
    jd_text: str
    temperature: float = 0.4
    resume_items: list[ResumeItem] = Field(default_factory=list)
    candidate_cards: Annotated[list[CandidateCard], operator.add] = Field(default_factory=list)
    processed: Annotated[bool, lambda old, new: bool(old) or bool(new)] = False
    email_pending: Annotated[bool, lambda old, new: bool(old) or bool(new)] = False
    message: str | None = None
