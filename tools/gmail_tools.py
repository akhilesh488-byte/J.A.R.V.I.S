"""
Gmail tools. Reading is straightforward; sending mail on your behalf
is a bigger trust step, so create_email_draft() creates a draft in
your Gmail account for you to review and send yourself rather than
sending automatically. If you decide later you want Jarvis to send
directly, that's a one-line change (users().drafts().send instead of
.create) — deliberately not wired that way by default.
"""

import base64
import sys
import os
from email.mime.text import MIMEText

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.google_auth import get_service


def list_recent_emails(count: int = 5) -> str:
    service = get_service("gmail", "v1")
    results = service.users().messages().list(userId="me", maxResults=count).execute()
    messages = results.get("messages", [])

    if not messages:
        return "You have no recent emails."

    summaries = []
    for msg in messages:
        detail = (
            service.users()
            .messages()
            .get(userId="me", id=msg["id"], format="metadata", metadataHeaders=["From", "Subject"])
            .execute()
        )
        headers = {h["name"]: h["value"] for h in detail["payload"]["headers"]}
        sender = headers.get("From", "Unknown sender")
        subject = headers.get("Subject", "(no subject)")
        summaries.append(f"From {sender}: {subject}")

    return "Your recent emails: " + "; ".join(summaries)


def create_email_draft(to: str, subject: str, body: str) -> str:
    service = get_service("gmail", "v1")

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
    return f"Drafted an email to {to} with subject '{subject}'. It's saved in your Drafts folder."
