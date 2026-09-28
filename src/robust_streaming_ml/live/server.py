"""Localhost HTTP server for the live-microphone demo (standard library only).

Only loopback addresses can be bound (127.0.0.1 by default). The server serves
the browser UI from package files and a small JSON API:

========  ======================================  =========================================
Method    Path                                    Meaning
========  ======================================  =========================================
GET       ``/``, ``/app.js``, ``/style.css``,     the UI (no external resources)
          ``/capture-worklet.js``
GET       ``/api/info``                           model identity, labels, contracts, limits
POST      ``/api/sessions``                       ``{"sample_rate_hz": R}`` -> new session
POST      ``/api/sessions/<id>/chunks?seq=<n>``   ``application/octet-stream`` f32le audio
POST      ``/api/sessions/<id>/stop``             flush; final decision
POST      ``/api/sessions/<id>/reset``            back to the initial state (seq 0)
DELETE    ``/api/sessions/<id>``                  discard the session
========  ======================================  =========================================

Requests whose ``Host`` header is not this server's loopback address, which
guards against DNS rebinding, are refused. So are writes whose ``Origin``
header is another site. No CORS headers are sent. Pages carry a strict
Content-Security-Policy. Audio is processed in memory and never written to
disk. Errors are JSON ``{"error": ...}`` with 400, 403, 404, 405, 409, 413, or 415.
"""

from __future__ import annotations

import ipaddress
import json
import secrets
import threading
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import parse_qs, urlsplit

from robust_streaming_ml.export.onnx_model import (
    LoadedOnnxModel,
    OnnxWindowBackend,
    create_session,
)
from robust_streaming_ml.live.resample import (
    BLOCK,
    KAISER_BETA,
    OUTPUT_RATE_HZ,
    SUPPORTED_INPUT_RATES,
    ZERO_CROSSINGS,
    ResampleError,
)
from robust_streaming_ml.live.session import (
    MAX_CHUNK_SECONDS,
    STABLE_RULE,
    AudioContractError,
    ChunkOrderError,
    LiveSession,
    SessionStateError,
)
from robust_streaming_ml.streaming.windowing import StreamingSpec

__all__ = ["LOOPBACK_HOSTS", "LiveDemoServer", "LiveServerError", "check_loopback"]

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
MAX_JSON_BYTES = 4096
# Largest request body read: 2 s at the highest supported rate, as float32.
MAX_BODY_BYTES = MAX_CHUNK_SECONDS * max(SUPPORTED_INPUT_RATES) * 4
_DRAIN_LIMIT = 64 * 1024 * 1024
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/capture-worklet.js": ("capture-worklet.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
}


class LiveServerError(RuntimeError):
    """The server cannot be started as requested."""


class _HttpError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def check_loopback(host: str) -> str:
    """Return ``host`` if it is a loopback name or address; refuse anything else."""
    if host == "localhost":
        return host
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    raise LiveServerError(
        f"refusing to bind {host!r}: the live demo only listens on loopback addresses "
        f"({', '.join(LOOPBACK_HOSTS)}), so it is never reachable from other machines"
    )


class LiveDemoServer:
    """Sessions plus a :class:`ThreadingHTTPServer` bound to a loopback address."""

    def __init__(
        self,
        model: LoadedOnnxModel,
        *,
        info: dict[str, Any],
        host: str = "127.0.0.1",
        port: int = 8765,
        intra_op_threads: int = 1,
        max_sessions: int = 4,
    ) -> None:
        self.host = check_loopback(host)
        self.model = model
        self.labels = model.labels
        self.spec = StreamingSpec.from_window(model.window)
        self.backend = OnnxWindowBackend(create_session(model, intra_op_threads=intra_op_threads))
        self.max_sessions = max_sessions
        self.sessions: OrderedDict[str, tuple[LiveSession, threading.Lock]] = OrderedDict()
        self.lock = threading.Lock()
        self.info = {
            **info,
            "labels": list(self.labels),
            "streaming": self.spec.describe(),
            "audio_contract": {
                "format": "f32le (little-endian IEEE float32), mono = mean of device channels",
                "supported_device_rates_hz": list(SUPPORTED_INPUT_RATES),
                "max_chunk_seconds": MAX_CHUNK_SECONDS,
                "ordering": "chunks carry seq = 0, 1, 2, ...; any other order is rejected",
                "conversion": (
                    f"Kaiser-windowed sinc resampler to {OUTPUT_RATE_HZ} Hz "
                    f"({ZERO_CROSSINGS} zero crossings, beta {KAISER_BETA}, unit DC gain per "
                    f"phase, {BLOCK}-sample blocks), then clip and round to 16-bit PCM"
                ),
            },
            "stable_rule": STABLE_RULE,
        }
        handler = type("Handler", (_Handler,), {"app": self})
        try:
            self.httpd = ThreadingHTTPServer((self.host, port), handler)
        except OSError as exc:
            raise LiveServerError(f"cannot listen on {self.host}:{port}: {exc}") from exc
        self.httpd.daemon_threads = True
        self.port = int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/"

    def allowed_hosts(self) -> set[str]:
        return {f"{h}:{self.port}" for h in ("127.0.0.1", "localhost", "[::1]")}

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        """Stop a server running :meth:`serve_forever` in another thread, then close it."""
        self.httpd.shutdown()
        self.httpd.server_close()

    def close(self) -> None:
        self.httpd.server_close()

    # --- sessions ---------------------------------------------------------------------

    def create(self, rate: int) -> str:
        session = LiveSession(self.backend, self.spec, self.labels, input_rate=rate)
        session_id = secrets.token_hex(8)
        with self.lock:
            while len(self.sessions) >= self.max_sessions:
                self.sessions.popitem(last=False)  # evict the least recently used
            self.sessions[session_id] = (session, threading.Lock())
        return session_id

    def get(self, session_id: str) -> tuple[LiveSession, threading.Lock]:
        with self.lock:
            if session_id not in self.sessions:
                raise _HttpError(HTTPStatus.NOT_FOUND, f"unknown session {session_id!r}")
            self.sessions.move_to_end(session_id)
            return self.sessions[session_id]

    def delete(self, session_id: str) -> None:
        with self.lock:
            if self.sessions.pop(session_id, None) is None:
                raise _HttpError(HTTPStatus.NOT_FOUND, f"unknown session {session_id!r}")


class _Handler(BaseHTTPRequestHandler):
    app: LiveDemoServer
    server_version = "rsml-live-demo"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        return  # quiet: about ten audio chunks per second

    # --- responses --------------------------------------------------------------------

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    # --- request checks ---------------------------------------------------------------

    def _check_host(self) -> None:
        if self.headers.get("Host", "") not in self.app.allowed_hosts():
            raise _HttpError(HTTPStatus.FORBIDDEN, "unexpected Host header")

    def _check_origin(self) -> None:
        origin = self.headers.get("Origin")
        allowed = {f"http://{h}" for h in self.app.allowed_hosts()}
        if origin is not None and origin not in allowed:
            raise _HttpError(HTTPStatus.FORBIDDEN, "cross-origin requests are not allowed")

    def _read_body(self) -> bytes:
        """Read the whole body first (bounded), so every error response is clean."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            self.close_connection = True
            raise _HttpError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from exc
        if length < 0:
            self.close_connection = True
            raise _HttpError(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
        if length > MAX_BODY_BYTES:
            if length <= _DRAIN_LIMIT:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 1 << 16))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            else:
                self.close_connection = True
            raise _HttpError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body over {MAX_BODY_BYTES} bytes"
            )
        return self.rfile.read(length) if length else b""

    @staticmethod
    def _limit(body: bytes, limit: int) -> bytes:
        if len(body) > limit:
            raise _HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body over {limit} bytes")
        return body

    def _content_type(self, expected: str) -> None:
        found = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if found != expected:
            raise _HttpError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE, f"Content-Type must be {expected}, got {found!r}"
            )

    # --- dispatch ---------------------------------------------------------------------

    def _handle(self, method: str) -> None:
        try:
            body = self._read_body() if method in ("POST", "DELETE") else b""
            self._check_host()
            if method in ("POST", "DELETE"):
                self._check_origin()
            parts = urlsplit(self.path)
            self._route(method, parts.path, parse_qs(parts.query), body)
        except _HttpError as exc:
            self._error(exc.status, str(exc))

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_DELETE(self) -> None:
        self._handle("DELETE")

    def _route(self, method: str, path: str, query: dict[str, list[str]], body: bytes) -> None:
        if path in STATIC or path == "/api/info":
            if method != "GET":
                raise _HttpError(HTTPStatus.METHOD_NOT_ALLOWED, f"{method} not allowed")
            if path == "/api/info":
                self._json(HTTPStatus.OK, self.app.info)
                return
            name, content_type = STATIC[path]
            package = resources.files("robust_streaming_ml.live")
            data = package.joinpath("static").joinpath(name).read_bytes()
            self._send(HTTPStatus.OK, data, content_type)
            return
        if path == "/api/sessions":
            if method != "POST":
                raise _HttpError(HTTPStatus.METHOD_NOT_ALLOWED, f"{method} not allowed")
            self._create(body)
            return
        segments = path.strip("/").split("/")
        if len(segments) >= 3 and segments[:2] == ["api", "sessions"]:
            session_id = segments[2]
            action = segments[3] if len(segments) == 4 else None
            if len(segments) > 4:
                raise _HttpError(HTTPStatus.NOT_FOUND, "not found")
            if action is None and method == "DELETE":
                self.app.delete(session_id)
                self._json(HTTPStatus.OK, {"deleted": session_id})
                return
            if method != "POST" or action not in ("chunks", "stop", "reset"):
                raise _HttpError(HTTPStatus.METHOD_NOT_ALLOWED, f"{method} {path} not allowed")
            session, lock = self.app.get(session_id)
            with lock:
                if action == "chunks":
                    self._chunk(session, query, body)
                elif action == "stop":
                    self._limit(body, MAX_JSON_BYTES)  # any small body is ignored
                    self._session_call(lambda: session.stop())
                else:
                    self._limit(body, MAX_JSON_BYTES)
                    session.reset()
                    self._json(HTTPStatus.OK, {"reset": True, "next_seq": 0})
            return
        raise _HttpError(HTTPStatus.NOT_FOUND, "not found")

    def _create(self, body: bytes) -> None:
        self._content_type("application/json")
        try:
            payload = json.loads(self._limit(body, MAX_JSON_BYTES) or b"{}")
            rate = payload["sample_rate_hz"]
        except (ValueError, KeyError, TypeError) as exc:
            raise _HttpError(
                HTTPStatus.BAD_REQUEST, 'body must be {"sample_rate_hz": <int>}'
            ) from exc
        if isinstance(rate, float) and rate.is_integer():
            rate = int(rate)  # browsers report AudioContext.sampleRate as a float
        if not isinstance(rate, int) or isinstance(rate, bool):
            raise _HttpError(HTTPStatus.BAD_REQUEST, "sample_rate_hz must be an integer")
        try:
            session_id = self.app.create(rate)
        except ResampleError as exc:
            raise _HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        self._json(
            HTTPStatus.CREATED,
            {"session": session_id, "sample_rate_hz": rate, "next_seq": 0},
        )

    def _chunk(self, session: LiveSession, query: dict[str, list[str]], body: bytes) -> None:
        self._content_type("application/octet-stream")
        values = query.get("seq", [])
        if len(values) != 1 or not values[0].isdigit():
            raise _HttpError(
                HTTPStatus.BAD_REQUEST, "query parameter seq=<non-negative int> required"
            )
        audio = self._limit(body, MAX_CHUNK_SECONDS * session.input_rate * 4)
        self._session_call(lambda: session.push(int(values[0]), audio))

    def _session_call(self, call: Any) -> None:
        try:
            result = call()
        except (AudioContractError, ResampleError) as exc:
            raise _HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        except (ChunkOrderError, SessionStateError) as exc:
            raise _HttpError(HTTPStatus.CONFLICT, str(exc)) from exc
        self._json(HTTPStatus.OK, result)
