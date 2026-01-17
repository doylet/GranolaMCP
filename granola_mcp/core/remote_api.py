"""Remote API client for GranolaMCP.

Granola's desktop app can fetch transcripts from the Granola backend even when the
local cache doesn't contain transcript segments. This module implements a minimal
standard-library-only client that mirrors the app's behavior:
- POST JSON to https://api.granola.ai/v1/<endpoint>
- Use Bearer access tokens stored locally by the desktop app

Notes on auth:
- Granola's Electron app stores WorkOS tokens in
  ~/Library/Application Support/Granola/supabase.json on macOS.
- The file contains a JSON-encoded string under "workos_tokens" with
  "access_token" and "refresh_token".

This client intentionally avoids logging or persisting any secret values.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


DEFAULT_API_BASE_URL = "https://api.granola.ai"
DEFAULT_SUPABASE_PATH_MAC = "~/Library/Application Support/Granola/supabase.json"
DEBUG_REMOTE_ENV = "GRANOLA_DEBUG_REMOTE"
CLIENT_VERSION_ENV = "GRANOLA_CLIENT_VERSION"
WORKSPACE_ID_ENV = "GRANOLA_WORKSPACE_ID"
DEVICE_ID_ENV = "GRANOLA_DEVICE_ID"


class GranolaRemoteError(RuntimeError):
    """Raised when a remote Granola API operation fails."""


def _debug_enabled() -> bool:
    value = os.environ.get(DEBUG_REMOTE_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(f"[granola-remote] {msg}", file=sys.stderr)


def _read_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise GranolaRemoteError(f"Expected JSON object in {path}")
    return data


def load_workos_tokens(
    supabase_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Load (access_token, refresh_token) from the local Granola app store.

    Returns (None, None) if tokens can't be found.
    """

    env_path = os.environ.get("GRANOLA_SUPABASE_PATH")
    candidate = supabase_path or env_path or DEFAULT_SUPABASE_PATH_MAC
    candidate = os.path.expanduser(candidate)

    if not os.path.exists(candidate):
        return None, None

    try:
        root = _read_json_file(candidate)
        workos_blob = root.get("workos_tokens")
        if not isinstance(workos_blob, str) or not workos_blob.strip():
            return None, None

        decoded = json.loads(workos_blob)
        if not isinstance(decoded, dict):
            return None, None

        access_token = decoded.get("access_token")
        refresh_token = decoded.get("refresh_token")

        if not isinstance(access_token, str) or not access_token.strip():
            access_token = None
        if not isinstance(refresh_token, str) or not refresh_token.strip():
            refresh_token = None

        return access_token, refresh_token
    except Exception:
        return None, None


@dataclass
class GranolaAPIClient:
    """Minimal Granola backend API client."""

    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    api_base_url: str = DEFAULT_API_BASE_URL
    timeout_seconds: float = 15.0
    client_version: Optional[str] = None
    workspace_id: Optional[str] = None
    device_id: Optional[str] = None

    @classmethod
    def from_local_app_data(
        cls,
        *,
        client_version: Optional[str] = None,
        workspace_id: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> "GranolaAPIClient":
        access_token, refresh_token = load_workos_tokens()
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            client_version=client_version or os.environ.get(CLIENT_VERSION_ENV),
            workspace_id=workspace_id or os.environ.get(WORKSPACE_ID_ENV),
            device_id=device_id or os.environ.get(DEVICE_ID_ENV),
        )

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "GranolaMCP/1.0",
        }

        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        if self.client_version and str(self.client_version).strip():
            headers["X-Client-Version"] = str(self.client_version).strip()
        else:
            # The Electron app always sends X-Client-Version; keep one present.
            headers["X-Client-Version"] = "GranolaMCP"

        if self.workspace_id and str(self.workspace_id).strip():
            headers["X-Granola-Workspace-Id"] = str(self.workspace_id).strip()

        if self.device_id and str(self.device_id).strip():
            headers["X-Granola-Device-Id"] = str(self.device_id).strip()

        return headers

    def _request_json(
        self,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        url = f"{self.api_base_url.rstrip('/')}/v1/{endpoint.lstrip('/')}"

        headers = self._build_headers()
        if extra_headers:
            for k, v in extra_headers.items():
                if v is None:
                    continue
                headers[str(k)] = str(v)

        body: Optional[bytes] = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            # Avoid leaking response bodies (could contain sensitive user data).
            # In debug mode, emit only a small, best-effort safe summary.
            if _debug_enabled():
                try:
                    raw_err = e.read(2048).decode("utf-8", errors="replace")
                    if raw_err:
                        try:
                            parsed = json.loads(raw_err)
                            if isinstance(parsed, dict):
                                msg = parsed.get("message") or parsed.get("error")
                                keys = sorted(parsed.keys())[:10]
                                if isinstance(msg, str) and msg.strip():
                                    _debug(f"{endpoint} error message: {msg[:200]}")
                                else:
                                    _debug(f"{endpoint} error keys: {keys}")
                            else:
                                _debug(f"{endpoint} error (non-dict JSON)")
                        except Exception:
                            _debug(f"{endpoint} error body (trunc): {raw_err[:200]}")
                except Exception:
                    pass

            if e.code == 401:
                _debug(f"HTTP 401 calling {endpoint}")
                raise GranolaRemoteError("Unauthorized (401) calling Granola API") from e
            _debug(f"HTTP {e.code} calling {endpoint}")
            raise GranolaRemoteError(f"Granola API call failed: {endpoint} (HTTP {e.code})") from e
        except urllib.error.URLError as e:
            _debug(f"Connection error calling {endpoint}: {type(e).__name__}")
            raise GranolaRemoteError(f"Granola API connection failed: {endpoint}") from e
        except json.JSONDecodeError as e:
            _debug(f"Invalid JSON from {endpoint}")
            raise GranolaRemoteError(f"Granola API returned invalid JSON for {endpoint}") from e

    def refresh_access_token(self) -> bool:
        """Attempt to refresh the access token using refresh-access-token.

        Returns True if the access token was updated in-memory.
        """
        if not self.refresh_token:
            return False

        response = None
        errors: list[Exception] = []

        payload_candidates = [
            {"refresh_token": self.refresh_token},
            {"refreshToken": self.refresh_token},
        ]

        for payload in payload_candidates:
            try:
                response = self._request_json("refresh-access-token", payload=payload)
                break
            except GranolaRemoteError as e:
                errors.append(e)
                continue

        if response is None:
            _ = errors
            return False

        if not isinstance(response, dict):
            return False

        new_access = response.get("access_token") or response.get("accessToken")
        new_refresh = response.get("refresh_token") or response.get("refreshToken")

        if isinstance(new_access, str) and new_access.strip():
            self.access_token = new_access
            if isinstance(new_refresh, str) and new_refresh.strip():
                self.refresh_token = new_refresh
            return True

        return False

    def get_document_transcript(self, document_id: str) -> Optional[Any]:
        """Fetch transcript payload for a document.

        Returns the transcript payload as either:
        - list[dict] transcript segments, or
        - a dict payload (caller can pass directly to Transcript)

        Returns None when no transcript is available.
        """

        if not document_id:
            return None

        payload_candidates = [
            {"document_id": document_id},
            {"documentId": document_id},
            {"id": document_id},
        ]

        last_error: Optional[Exception] = None
        for payload in payload_candidates:
            try:
                response = self._request_json("get-document-transcript", payload=payload)
            except GranolaRemoteError as e:
                last_error = e
                # If token expired, try refreshing once.
                if "Unauthorized" in str(e) and self.refresh_access_token():
                    try:
                        response = self._request_json("get-document-transcript", payload=payload)
                    except GranolaRemoteError as e2:
                        last_error = e2
                        continue
                else:
                    continue

            if response is None:
                return None

            # Common shapes:
            # - {"transcript": [...segments...]}
            # - {"data": {"transcript": [...]}}
            # - [...segments...]
            if isinstance(response, list):
                return response

            if isinstance(response, dict):
                if "transcript" in response:
                    return response.get("transcript")
                data = response.get("data")
                if isinstance(data, dict) and "transcript" in data:
                    return data.get("transcript")

                # Some APIs nest under "data" as a list.
                if isinstance(data, list):
                    return data

                # Some API responses may already be the transcript payload.
                return response

            # Unexpected type; treat as unavailable.
            return None

        # If every attempt failed, surface None (best-effort).
        # Callers should not consider this fatal.
        _ = last_error
        return None
