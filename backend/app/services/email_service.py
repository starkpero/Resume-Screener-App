from __future__ import annotations

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from app.core.settings import get_settings, load_config


async def send_outreach_email(*, to_email: str, subject: str, body: str, from_email: str | None = None) -> dict:
    settings = get_settings()
    config = load_config()
    sender_email = from_email or config["sender_email"]

    # Validate API key is configured
    if not settings.sendgrid_api_key or not settings.sendgrid_api_key.startswith("SG."):
        raise ValueError(
            "SendGrid API key not properly configured. "
            "Please verify SENDGRID_API_KEY in your .env file and ensure it's valid (should start with 'SG.')."
        )

    message = Mail(
        from_email=sender_email,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body,
    )

    try:
        client = SendGridAPIClient(settings.sendgrid_api_key)
        response = client.send(message)
        return {
            "status_code": response.status_code,
            "body": response.body.decode("utf-8", errors="ignore") if hasattr(response.body, "decode") else str(response.body),
            "headers": dict(response.headers),
        }
    except Exception as e:
        error_message = str(e)
        if "403" in error_message or "Forbidden" in error_message:
            raise ValueError(
                f"SendGrid API Error 403 Forbidden: Your API key is invalid, revoked, or doesn't have permission to send emails. "
                f"Please verify your SendGrid API key has the 'mail.send' permission and that {sender_email} is verified as a sender."
            ) from e
        raise
