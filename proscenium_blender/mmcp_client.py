"""Minimal MMCP HTTP client.

Stdlib-only — Blender ships its own Python interpreter and adding a third-party
``requests`` dependency makes installation finicky. ``urllib`` is enough.

Public surface:
  * ``get_server_url()`` — read the configured base URL from addon prefs.
  * ``MmcpError`` — typed exception carrying the MMCP error envelope.
  * ``MmcpClient.capabilities()`` — cached ``GET /capabilities``.
  * ``MmcpClient.generate(req)`` — synchronous ``POST /generate`` returning a
    parsed glTF JSON document. Handles the optional ``202 Accepted`` async
    poll loop transparently when the model declares ``supports_async``.

Threading: the addon calls these from a worker thread; nothing here touches
``bpy`` so it's safe.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import bpy


# ---------------------------------------------------------------------------
# Preferences plumbing
# ---------------------------------------------------------------------------

def get_server_url() -> str:
    """Return the bare configured server URL (no trailing slash, no path prefix).

    For Animatica Cloud this is ``https://api.animatica.ai`` — the host
    that owns ``/auth/*``, ``/account``, etc. For self-hosted users
    it's the override URL they typed.

    Use ``get_mmcp_url()`` if you need the base for MMCP requests
    specifically; on cloud those live one path-segment deeper, behind
    the auth proxy.
    """
    from .properties import CLOUD_API_URL
    addon = bpy.context.preferences.addons.get(__package__)
    if addon is None:
        return CLOUD_API_URL
    prefs = addon.preferences
    if getattr(prefs, "self_hosted", False):
        url = (prefs.server_url or "").strip()
        return (url or "http://localhost:8000").rstrip("/")
    return CLOUD_API_URL.rstrip("/")


def get_mmcp_url() -> str:
    """Return the base URL for MMCP requests (``/capabilities``, ``/generate``).

    On Animatica Cloud the MMCP server sits behind an auth/quota proxy
    at ``api.animatica.ai/mmcp``; on self-hosted setups the MMCP server
    *is* the user's server, so there's no path prefix. Either way the
    plugin's ``MmcpClient`` should be constructed with this URL — it
    appends ``/capabilities`` etc. directly.
    """
    from .properties import CLOUD_API_URL
    addon = bpy.context.preferences.addons.get(__package__)
    if addon is None:
        return f"{CLOUD_API_URL.rstrip('/')}/mmcp"
    prefs = addon.preferences
    if getattr(prefs, "self_hosted", False):
        url = (prefs.server_url or "").strip()
        return (url or "http://localhost:8000").rstrip("/")
    return f"{CLOUD_API_URL.rstrip('/')}/mmcp"


# ---------------------------------------------------------------------------
# Auth — Animatica Cloud only. Self-hosted servers ignore the Authorization
# header. Auth is NOT part of the MMCP protocol; the cloud's proxy in front
# of /generate is what consumes the token. The plugin treats it as
# "attach if present, prompt to sign in on 401".
# ---------------------------------------------------------------------------

def _addon_prefs():
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


def get_access_token() -> str:
    p = _addon_prefs()
    return ((getattr(p, "access_token", "") or "").strip()) if p else ""


def get_refresh_token() -> str:
    p = _addon_prefs()
    return ((getattr(p, "refresh_token", "") or "").strip()) if p else ""


def _auth_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Standard request headers + Bearer token if signed in."""
    hdrs: dict[str, str] = {}
    if extra:
        hdrs.update(extra)
    token = get_access_token()
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    return hdrs


def sign_in(email: str, password: str) -> dict[str, Any]:
    """POST /auth/login on the cloud auth proxy. Stores tokens on AddonPreferences.

    The /auth/* endpoints live at the bare host (``api.animatica.ai/auth/login``),
    not under ``/mmcp/``, so this always uses ``get_server_url()`` rather
    than the MMCP base. Self-hosted setups don't sign in at all.
    """
    base_url = get_server_url()
    body = json.dumps({"email": email, "password": password, "client": "blender"}).encode()
    req = Request(
        f"{base_url}/auth/login",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except HTTPError as exc:
        raise MmcpError.from_response(exc.code, exc.read()) from exc
    except URLError as exc:
        raise MmcpError(
            code="model_unavailable",
            message=f"cannot reach {base_url}: {exc.reason}",
        ) from exc

    p = _addon_prefs()
    if p is not None:
        p.access_token = data.get("access_token", "")
        p.refresh_token = data.get("refresh_token", "")
        p.email = data.get("email", email)
        p.tier = data.get("tier", "")
    return data


def sign_out() -> None:
    """Forget cached tokens. The server-side session may still be valid;
    self-clear is enough for the plugin's purposes."""
    p = _addon_prefs()
    if p is not None:
        p.access_token = ""
        p.refresh_token = ""
        p.email = ""
        p.tier = ""


def refresh_access_token() -> bool:
    """POST /auth/refresh. Returns True on success and updates prefs.

    Always hits the bare cloud auth host (``get_server_url()``), not the
    MMCP base — ``/auth/refresh`` lives outside the ``/mmcp/`` namespace.
    """
    rt = get_refresh_token()
    if not rt:
        return False
    base_url = get_server_url()
    req = Request(
        f"{base_url}/auth/refresh",
        data=json.dumps({"refresh_token": rt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return False
    p = _addon_prefs()
    if p is None:
        return False
    new_access = data.get("access_token", "")
    if not new_access:
        return False
    p.access_token = new_access
    p.refresh_token = data.get("refresh_token", rt)
    if data.get("tier"):
        p.tier = data["tier"]
    return True


# ---------------------------------------------------------------------------
# Process-wide capabilities cache.
#
# The Connect operator writes here once on success; the EnumProperty items
# callback in ``properties.py`` reads from here. Strings in the items list
# must stay alive for as long as Blender holds references to them, hence the
# module-level list.
# ---------------------------------------------------------------------------

_CAPABILITIES: dict[str, Any] | None = None
_MODEL_ITEMS: list[tuple[str, str, str]] = []
_LAST_ERROR: str = ""


def cached_capabilities() -> dict[str, Any] | None:
    return _CAPABILITIES


def cached_model_items() -> list[tuple[str, str, str]]:
    """Return a static list of ``(id, label, description)`` tuples for use
    as ``EnumProperty(items=...)`` values.

    Returns at least one entry so Blender always has a valid default; when no
    capabilities have been fetched, returns a sentinel item that's clearly
    not a real model id.
    """
    if _MODEL_ITEMS:
        return _MODEL_ITEMS
    return [("", "(connect to discover models)", "")]


def cached_model(model_id: str) -> dict[str, Any] | None:
    if _CAPABILITIES is None:
        return None
    for m in _CAPABILITIES.get("models", []):
        if m.get("id") == model_id:
            return m
    return None


def store_capabilities(caps: dict[str, Any]) -> None:
    """Replace the cache and rebuild the EnumProperty items list."""
    global _CAPABILITIES, _MODEL_ITEMS, _LAST_ERROR
    _CAPABILITIES = caps
    _LAST_ERROR = ""
    items: list[tuple[str, str, str]] = []
    for m in caps.get("models", []):
        mid = m.get("id", "")
        if not mid:
            continue
        fps = m.get("fps", "?")
        joints = len(m.get("canonical_skeleton", {}).get("joints", []))
        items.append((mid, mid, f"{joints} joints @ {fps} fps"))
    _MODEL_ITEMS = items


def clear_capabilities(error: str = "") -> None:
    global _CAPABILITIES, _MODEL_ITEMS, _LAST_ERROR
    _CAPABILITIES = None
    _MODEL_ITEMS = []
    _LAST_ERROR = error


def last_connection_error() -> str:
    return _LAST_ERROR


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class MmcpError(Exception):
    """Wraps the MMCP error envelope.

    ``code`` is the protocol code (``unknown_model``, ``constraint_conflict``,
    …); ``status`` is the HTTP status; ``details`` is the optional dict from
    the envelope. The string form is ``"<code>: <message>"`` so it's safe to
    surface in a Blender ``self.report({'ERROR'}, str(exc))`` call.
    """

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status = status
        self.details = details or {}

    @classmethod
    def from_response(cls, status: int, body: bytes) -> "MmcpError":
        """Best-effort parse of a non-2xx response into an MmcpError."""
        try:
            payload = json.loads(body)
            err = payload.get("error", {}) if isinstance(payload, dict) else {}
            return cls(
                code=err.get("code", "internal_error"),
                message=err.get("message") or body[:200].decode("utf-8", errors="replace"),
                status=status,
                details=err.get("details") or {},
            )
        except Exception:
            return cls(
                code="internal_error",
                message=f"non-JSON error response (HTTP {status})",
                status=status,
            )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class MmcpClient:
    """Single-purpose MMCP client.

    Construct with a base URL (typically from ``get_server_url()``). The
    capabilities response is cached on first read and can be refreshed by
    calling ``capabilities(refresh=True)``.
    """

    DEFAULT_TIMEOUT_SECONDS = 600  # generation can take minutes

    def __init__(self, base_url: str | None = None, *, timeout: float | None = None):
        self.base_url = (base_url or get_server_url()).rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT_SECONDS
        self._caps: dict[str, Any] | None = None

    # --- Capabilities ------------------------------------------------------

    def capabilities(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._caps is None or refresh:
            self._caps = self._get_json("/capabilities")
        return self._caps

    def model(self, model_id: str) -> dict[str, Any]:
        for m in self.capabilities().get("models", []):
            if m.get("id") == model_id:
                return m
        available = [m.get("id") for m in self.capabilities().get("models", [])]
        raise MmcpError(
            code="unknown_model",
            message=f"model {model_id!r} is not in /capabilities",
            details={"available_models": available},
        )

    # --- Generation --------------------------------------------------------

    def generate(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """POST a GenerateRequest. Returns the parsed glTF JSON document.

        Handles both sync (200) and async (202 + Location) responses
        transparently. Attaches an `Authorization: Bearer` header when a
        cloud session token is set; on a 401 we attempt one silent token
        refresh + retry before raising.
        """
        body = json.dumps(request_body).encode("utf-8")

        def _post():
            req = Request(
                f"{self.base_url}/generate",
                data=body,
                headers=_auth_headers({
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept":       "model/gltf+json",
                }),
            )
            return urlopen(req, timeout=self.timeout)

        try:
            try:
                resp = _post()
            except HTTPError as exc:
                # Refresh-and-retry once on 401 (auth-proxy session expired).
                if exc.code == 401 and refresh_access_token():
                    resp = _post()
                else:
                    raise
            with resp:
                if resp.status == 200:
                    return json.loads(resp.read())
                if resp.status == 202:
                    location = resp.headers.get("Location") or ""
                    retry_after = float(resp.headers.get("Retry-After") or "2")
                    return self._poll_job(location, retry_after)
                raise MmcpError.from_response(resp.status, resp.read())
        except HTTPError as exc:
            raise MmcpError.from_response(exc.code, exc.read()) from exc
        except URLError as exc:
            raise MmcpError(
                code="model_unavailable",
                message=f"cannot reach MMCP server at {self.base_url}: {exc.reason}",
            ) from exc

    def _poll_job(self, location: str, retry_after: float) -> dict[str, Any]:
        if not location.startswith("/"):
            # Defensive: if the server returned a fully-qualified URL, strip
            # to the path so we still hit our base_url.
            location = "/" + location.split("/", 3)[-1]
        url = f"{self.base_url}{location}"
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            time.sleep(max(retry_after, 0.5))
            try:
                req = Request(url, headers=_auth_headers())
                with urlopen(req, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read())
                    if resp.status == 202:
                        retry_after = float(resp.headers.get("Retry-After") or retry_after)
                        continue
                    raise MmcpError.from_response(resp.status, resp.read())
            except HTTPError as exc:
                raise MmcpError.from_response(exc.code, exc.read()) from exc
        raise MmcpError(code="timeout", message=f"async job at {url} did not complete in {self.timeout}s")

    # --- Internal HTTP -----------------------------------------------------

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"

        def _get():
            req = Request(url, headers=_auth_headers())
            return urlopen(req, timeout=self.timeout)

        try:
            try:
                resp = _get()
            except HTTPError as exc:
                if exc.code == 401 and refresh_access_token():
                    resp = _get()
                else:
                    raise
            with resp:
                if resp.status != 200:
                    raise MmcpError.from_response(resp.status, resp.read())
                return json.loads(resp.read())
        except HTTPError as exc:
            raise MmcpError.from_response(exc.code, exc.read()) from exc
        except URLError as exc:
            raise MmcpError(
                code="model_unavailable",
                message=f"cannot reach MMCP server at {self.base_url}: {exc.reason}",
            ) from exc
