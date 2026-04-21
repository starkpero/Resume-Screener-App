from __future__ import annotations

import asyncio
import ipaddress
import time
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.settings import get_settings, load_config
from app.graph.screening_graph import run_analysis
from app.models.schemas import CandidateResults, SendEmailRequest
from app.services.email_service import send_outreach_email
from app.services.resume_loader import load_resume_text_from_plain_text, load_resume_text_from_upload

"""
Resume Screening API - FastAPI Backend

This module provides REST API endpoints for the HR resume screening application.
It handles:
1. Resume analysis against job descriptions (with AI-powered semantic matching)
2. Email sending for candidate outreach
3. Rate limiting to prevent abuse
4. CORS configuration for frontend integration
"""

app = FastAPI(title="Recruiter Co-Pilot API", version="1.0.0")

settings = get_settings()
config = load_config()
cors_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
# Use provided origins, or allow all origins if empty (for development/debugging)
allow_origins = cors_origins if cors_origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting tracking dictionary
# Format: { client_key: { "count": int, "window_start": float } }
RATE_LIMITS: dict[str, dict[str, float | int]] = {}
RATE_LIMIT_MAX = int(config.get("request_limits", {}).get("analysis_requests_per_user", 2))
RATE_LIMIT_WINDOW_HOURS = int(config.get("request_limits", {}).get("window_hours", 24))
RATE_LIMIT_WINDOW_SECONDS = RATE_LIMIT_WINDOW_HOURS * 3600


def _client_key(request: Request) -> str:
    """
    Extract client identifier for rate limiting.
    
    Attempts to identify clients in this order:
    1. x-client-id header (for explicit client identification)
    2. x-forwarded-for header (for proxied requests)
    3. request.client.host (direct connection IP)
    4. "anonymous" fallback
    """
    header_key = request.headers.get("x-client-id")
    if header_key:
        return header_key.strip()
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        try:
            return str(ipaddress.ip_address(request.client.host))
        except Exception:
            return request.client.host
    return "anonymous"


def _enforce_rate_limit(key: str) -> int:
    """
    Enforce per-client rate limiting on analysis requests.
    
    Algorithm:
    1. Check if client has exceeded limit in current window
    2. If window has expired, reset count
    3. If limit reached, raise 429 Too Many Requests
    4. Otherwise, increment counter and return remaining requests
    
    Returns: Number of requests remaining in current window
    Raises: HTTPException with 429 status if limit exceeded
    """
    now = time.time()
    state = RATE_LIMITS.get(key)
    if not state or now - float(state["window_start"]) > RATE_LIMIT_WINDOW_SECONDS:
        RATE_LIMITS[key] = {"count": 0, "window_start": now}
        state = RATE_LIMITS[key]

    count = int(state["count"])
    if count >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Analysis request limit reached: {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW_HOURS} hours.",
            headers={"X-RateLimit-Limit": str(RATE_LIMIT_MAX), "X-RateLimit-Remaining": "0"},
        )
    state["count"] = count + 1
    return max(RATE_LIMIT_MAX - int(state["count"]), 0)


@app.get("/api/health")
async def health():
    """
    Health check endpoint.
    
    Returns: {"status": "ok"} when the API is running
    Used by: Load balancers and monitoring systems to verify service availability
    """
    return {"status": "ok"}


@app.post("/api/analyze", response_model=CandidateResults)
async def analyze_resume(
    request: Request,
    jd_text: str = Form(...),
    temperature: float = Form(0.4),
    resume_text: str | None = Form(None),
    resume_file: UploadFile | None = File(None),
):
    """
    Main resume analysis endpoint.
    
    This endpoint is the core of the application. It:
    1. Accepts a job description and resume(s)
    2. Enforces rate limiting per client
    3. Parses resume(s) - supports PDF and plain text formats
    4. Runs the LangGraph analysis pipeline
    5. Returns structured candidate cards with match scores and emails
    
    Parameters:
        jd_text (str): Job description text to match against
        temperature (float): LLM temperature for email tone (0.0-1.0)
            - 0.4: Default balanced tone
            - <0.4: More formal/professional
            - >0.4: More warm/enthusiastic
        resume_text (str, optional): Raw resume text (alternative to file)
        resume_file (UploadFile, optional): PDF or text file upload
    
    Returns:
        CandidateResults: Aggregated results with:
        - candidate_cards: List of analyzed candidates
        - processed: Whether analysis completed
        - email_pending: Whether outreach is ready
        - request_count_remaining: Calls left in rate limit window
    
    Raises:
        400: Missing resume or empty content
        429: Rate limit exceeded
    """
    key = _client_key(request)
    remaining = _enforce_rate_limit(key)

    if not resume_text and resume_file is None:
        raise HTTPException(status_code=400, detail="Provide resume_text or resume_file.")

    if resume_file is not None:
        parsed_resume_text, file_name = await load_resume_text_from_upload(resume_file)
    else:
        parsed_resume_text, file_name = load_resume_text_from_plain_text(resume_text or "")

    if not parsed_resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume content is empty.")

    # Run analysis in thread pool to avoid blocking event loop
    # LLM API calls and vector database operations are I/O bound
    result = await asyncio.to_thread(
        run_analysis,
        jd_text,
        [{"resume_text": parsed_resume_text, "file_name": file_name}],
        float(temperature),
    )

    return CandidateResults(
        candidate_cards=result.get("candidate_cards", []),
        processed=bool(result.get("processed", False)),
        email_pending=bool(result.get("email_pending", False)),
        request_count_remaining=remaining,
        message="Analysis complete",
    )


@app.post("/api/send-email")
async def send_email(payload: SendEmailRequest):
    """
    Send personalized outreach email to candidate.
    
    This endpoint allows HR teams to send the AI-generated email drafts.
    
    Process:
    1. Receives candidate contact info and email content
    2. Uses SendGrid API to send the email
    3. Returns delivery confirmation
    
    Parameters:
        candidate_email (str): Recipient email address
        candidate_name (str, optional): Name for personalization
        job_title (str): Position title
        subject (str, optional): Email subject line
        body (str, optional): Email body
    
    Returns:
        {"sent": True, "provider_result": {...}}: Delivery status and SendGrid response
    
    Raises:
        400: SendGrid configuration error or invalid credentials
        500: Email delivery failed
    """
    try:
        subject = payload.subject or f"Next Steps for {payload.job_title} position"
        body = payload.body or f"Hi {payload.candidate_name or 'there'},\n\nWe loved your profile and would like to continue the conversation.\n\nBest,\nRecruiting Team"
        result = await send_outreach_email(
            to_email=payload.candidate_email,
            subject=subject,
            body=body,
        )
        return {"sent": True, "provider_result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send email: {str(e)}. Please check your SendGrid configuration and API key.",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    """
    Global HTTP exception handler for consistent error responses.
    
    Returns all HTTP errors in a standard JSON format with proper headers.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers or None,
    )
