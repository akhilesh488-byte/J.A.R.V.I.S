import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.google_auth import get_service


def list_recent_files(count: int = 5) -> str:
    service = get_service("drive", "v3")
    results = (
        service.files()
        .list(pageSize=count, orderBy="modifiedTime desc", fields="files(name, modifiedTime)")
        .execute()
    )
    files = results.get("files", [])

    if not files:
        return "No recent files found in your Drive."

    summaries = [f["name"] for f in files]
    return "Your recent Drive files: " + "; ".join(summaries)
