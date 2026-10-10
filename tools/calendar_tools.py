import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.google_auth import get_service


def list_upcoming_events(count: int = 5) -> str:
    service = get_service("calendar", "v3")
    now = datetime.datetime.utcnow().isoformat() + "Z"

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=count,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    if not events:
        return "You have no upcoming events."

    summaries = []
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        summaries.append(f"{event.get('summary', 'Untitled')} at {start}")

    return "Your upcoming events: " + "; ".join(summaries)


def create_event(summary: str, start_time: str, end_time: str, description: str = "") -> str:
    """
    start_time / end_time should be ISO 8601, e.g. '2026-09-10T15:00:00'.
    The calling LLM is responsible for converting spoken times like
    "tomorrow at 3pm" into this format before calling the tool.
    """
    service = get_service("calendar", "v3")

    event_body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_time},
        "end": {"dateTime": end_time},
    }

    service.events().insert(calendarId="primary", body=event_body).execute()
    return f"Scheduled '{summary}' starting {start_time}."
