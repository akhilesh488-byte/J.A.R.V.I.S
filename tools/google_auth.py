"""
Shared OAuth flow for Gmail, Calendar, and Drive. All three Google
tools use the same credentials/token files and scopes, so the login
flow only needs to run once.

First-time setup (see README for full steps):
1. Create a project in Google Cloud Console, enable the Gmail,
   Calendar, and Drive APIs.
2. Create an OAuth client ID of type "Desktop app", download it as
   credentials.json, and put it in the jarvis/ root.
3. First time any Google tool runs, a browser window opens for you to
   log in and grant access. After that, token.json is saved and reused
   automatically (and refreshed as needed) — no repeated logins.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_service_cache = {}


def get_credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.exists(config.GOOGLE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            config.GOOGLE_TOKEN_FILE, config.GOOGLE_SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(config.GOOGLE_CREDENTIALS_FILE):
                raise RuntimeError(
                    "Missing credentials.json — download it from Google Cloud "
                    "Console (OAuth client ID, Desktop app type) and place it "
                    "in the jarvis/ root. See README for steps."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                config.GOOGLE_CREDENTIALS_FILE, config.GOOGLE_SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(config.GOOGLE_TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return creds


def get_service(api_name: str, api_version: str):
    """Returns a cached googleapiclient service, e.g. get_service('gmail', 'v1')."""
    key = (api_name, api_version)
    if key not in _service_cache:
        from googleapiclient.discovery import build

        _service_cache[key] = build(api_name, api_version, credentials=get_credentials())
    return _service_cache[key]
