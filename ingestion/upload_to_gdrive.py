"""
Upload Sales Collateral files to Google Drive and generate URL mapping.
=========================================================================

Usage:
    pip install google-api-python-client google-auth
    python upload_to_gdrive.py

Requires:
    - chatbot-490012-fdd706287009.json in the same directory (or set SERVICE_ACCOUNT_FILE env var)
    - Google Drive folder shared with the service account email as Editor

Output:
    - gdrive_urls.json — mapping of { "filename": "drive_view_url" }
"""

import json
import os
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ── Configuration ────────────────────────────────────────────────────────────

OAUTH_CLIENT_FILE = os.getenv(
    "OAUTH_CLIENT_FILE",
    str(Path(__file__).parent / "oauth_client.json"),
)
TOKEN_FILE = str(Path(__file__).parent / "token.json")
SCOPES = ["https://www.googleapis.com/auth/drive"]

# Your Google Drive folder ID
DRIVE_FOLDER_ID = "1HREu0XhcESu41SE0aMNPZ_3i4ZO257-I"

# Path to your Sales Collateral folder
DATA_DIR = Path(os.getenv(
    "DATA_DIR",
    r"C:\Users\janee\OneDrive\文档\chatboit\Sales Collateral",
))

# Extensions to upload
UPLOAD_EXTENSIONS = {".pdf", ".pptx", ".docx"}

# Output mapping file
OUTPUT_FILE = Path(__file__).parent / "gdrive_urls.json"

# MIME types
MIME_MAP = {
    ".pdf": "application/pdf",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# ── Google Drive client ──────────────────────────────────────────────────────

def get_drive_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CLIENT_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds)


def list_existing_files(service, folder_id: str) -> dict:
    """Get all files already in the Drive folder → {filename: file_id}."""
    existing = {}
    page_token = None

    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name)",
            pageSize=100,
            pageToken=page_token,
        ).execute()

        for f in resp.get("files", []):
            existing[f["name"]] = f["id"]

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return existing


def upload_file(service, file_path: Path, folder_id: str) -> str:
    """Upload a single file to Drive. Returns the file ID."""
    ext = file_path.suffix.lower()
    mime = MIME_MAP.get(ext, "application/octet-stream")

    metadata = {
        "name": file_path.name,
        "parents": [folder_id],
    }

    media = MediaFileUpload(str(file_path), mimetype=mime, resumable=True)

    uploaded = service.files().create(
        body=metadata,
        media_body=media,
        fields="id",
    ).execute()

    return uploaded["id"]


def make_public(service, file_id: str) -> None:
    """Make a file viewable by anyone with the link."""
    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
    ).execute()


def drive_view_url(file_id: str) -> str:
    """Generate a direct view/download URL."""
    return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Data dir  : {DATA_DIR}")
    print(f"Folder ID : {DRIVE_FOLDER_ID}")
    print(f"Output    : {OUTPUT_FILE}\n")

    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        return

    service = get_drive_service()

    # Find all uploadable files
    all_files = sorted(
        f for f in DATA_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in UPLOAD_EXTENSIONS
    )
    print(f"Found {len(all_files)} files to upload\n")

    # Check what's already uploaded (skip duplicates)
    existing = list_existing_files(service, DRIVE_FOLDER_ID)
    print(f"Already in Drive: {len(existing)} files\n")

    # Load existing mapping if present
    url_map = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            url_map = json.load(f)

    # Upload
    for i, file_path in enumerate(all_files, 1):
        name = file_path.name

        if name in existing:
            file_id = existing[name]
            print(f"  [{i}/{len(all_files)}] SKIP (exists) {name}")
        else:
            print(f"  [{i}/{len(all_files)}] Uploading {name} ... ", end="", flush=True)
            try:
                file_id = upload_file(service, file_path, DRIVE_FOLDER_ID)
                make_public(service, file_id)
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")
                continue

        url_map[name] = drive_view_url(file_id)

    # Save mapping
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(url_map, f, indent=2, ensure_ascii=False)

    print(f"\nComplete! {len(url_map)} files mapped → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
