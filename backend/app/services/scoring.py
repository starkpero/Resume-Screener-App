"""Scoring and matching algorithms for resume-to-job matching.

Key algorithms:
- weighted_match_percentage: Combines multiple scoring dimensions
- build_jd_query: Extracts searchable keywords from job description
- role_title: Extracts job title for email personalization
"""
from __future__ import annotations

from typing import Any

from app.core.settings import load_config
from app.models.schemas import MatchAnalysis, StructuredProfile


def weighted_match_percentage(match_analysis: MatchAnalysis, *, config: dict[str, Any] | None = None) -> tuple[float, dict[str, float]]:
    """
    Calculate overall match percentage using weighted scoring.
    
    Algorithm:
    1. Load configurable weights from config.json (default: 70% tech, 20% experience, 10% seniority)
    2. Normalize weights to sum = 1.0 to handle missing config values
    3. Compute weighted average across three dimensions:
       - Technical Skills Score: Core competency alignment (70% default)
       - Experience Years Score: Seniority level match (20% default)
       - Seniority Alignment: Career level fit (10% default)
    4. Round to 2 decimal places
    
    This approach allows HR teams to customize scoring priorities via config.json
    
    Args:
        match_analysis (MatchAnalysis): Individual scores from semantic_match node
        config (dict, optional): Override configuration (defaults to config.json)
    
    Returns:
        tuple: (overall_percentage: float, breakdown: dict)
            - overall_percentage: 0-100 weighted score
            - breakdown: Individual component scores for transparency
    """
    cfg = config or load_config()
    weights = cfg.get("scoring_weights", {})
    technical = float(weights.get("technical_skills", 0.7))
    experience = float(weights.get("experience_years", 0.2))
    seniority = float(weights.get("seniority_alignment", 0.1))
    total = technical + experience + seniority or 1.0

    score = (
        match_analysis.technical_skills_score * technical
        + match_analysis.experience_years_score * experience
        + match_analysis.seniority_alignment_score * seniority
    ) / total

    breakdown = {
        "technical_skills": round(match_analysis.technical_skills_score, 2),
        "experience_years": round(match_analysis.experience_years_score, 2),
        "seniority_alignment": round(match_analysis.seniority_alignment_score, 2),
    }
    return round(score, 2), breakdown


def build_jd_query(jd: StructuredProfile) -> str:
    """
    Build a search query from job description for vector similarity search.
    
    Process:
    1. Aggregate all relevant keywords and phrases from JD:
       - Core skills (hard requirements)
       - Must-have requirements
       - Nice-to-have requirements
       - Soft skills
       - Job title
       - Summary/description
    2. Join with " | " separator for semantic search query
    
    This query is used to find the most relevant resume sections in Chroma
    vector database via semantic similarity.
    
    Args:
        jd (StructuredProfile): Structured job description
    
    Returns:
        str: Query string for vector database search
    """
    parts: list[str] = []
    parts.extend(jd.skills)
    parts.extend(jd.must_have)
    parts.extend(jd.nice_to_have)
    parts.extend(jd.soft_skills)
    if jd.title:
        parts.append(jd.title)
    if jd.summary:
        parts.append(jd.summary)
    return " | ".join(part for part in parts if part)


def role_title(jd: StructuredProfile) -> str:
    """
    Extract job title from structured job description.
    
    Fallback: Uses summary if title is not available.
    
    Args:
        jd (StructuredProfile): Structured job description
    
    Returns:
        str: Job title for personalization
    """
    return jd.title or "the role"
