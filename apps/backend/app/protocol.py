"""Python mirror of the `@fatiguevision/contracts` WebSocket protocol.

The TypeScript package under packages/contracts is the SINGLE source of truth
for the wire; the Python backend deliberately does NOT import it. Every
literal and shape below reproduces the contract strings so the backend emits
exactly the same envelope, kinds and payloads (verified by the contract
smoke test).
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass
from typing import cast

PROTOCOL_VERSION = "1"
WS_PATH = "/ws"


class MessageKind:
    """Mirror of the ``MessageKind`` const object from contracts/constants.ts."""

    VISION_STATUS = "vision.status"
    FATIGUE_ASSESSMENT = "fatigue.assessment"
    BIOMETRIC_EVENT = "biometric.event"
    FACE_OVERLAY = "face.overlay"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    CONTROL_START = "control.start"
    CONTROL_STOP = "control.stop"
    FRAME_PROVIDE = "frame.provide"


# Stable server error codes (contracts/models.ts ErrorPayload.code).
ERROR_CAMERA_UNAVAILABLE = "camera.unavailable"
ERROR_PIPELINE = "pipeline.error"
ERROR_INFERENCE = "inference.error"
ERROR_PROTOCOL = "protocol.error"

# Kinds the client MAY SEND (contracts/messages.ts ClientToServerMessage).
CLIENT_KINDS = frozenset(
    {
        MessageKind.PING,
        MessageKind.CONTROL_START,
        MessageKind.CONTROL_STOP,
        MessageKind.FRAME_PROVIDE,
    }
)

# Envelope keys mandated by contracts/messages.ts MessageEnvelope.
ENVELOPE_KEYS = frozenset(
    {"messageId", "timestamp", "protocolVersion", "kind", "payload"}
)


def build_envelope(kind: str, payload: dict[str, object]) -> dict[str, object]:
    """Build a MessageEnvelope-shaped dict for the given kind and payload."""
    return {
        "messageId": uuid.uuid4().hex,
        "timestamp": time.time(),
        "protocolVersion": PROTOCOL_VERSION,
        "kind": kind,
        "payload": payload,
    }


def serialize(message: dict[str, object]) -> str:
    """Serialize an envelope to its wire JSON (compact, UTF-8)."""
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


class ProtocolError(Exception):
    """Raised when an incoming client message violates the wire contract."""


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """A successfully parsed client message."""

    message_id: str
    kind: str
    payload: dict[str, object]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_frame_provide(payload: dict[str, object]) -> None:
    """Validate a ``frame.provide`` payload against the FrameProvidePayload shape.

    FrameProvidePayload (contracts/models.ts): ``frameId`` (int, emitter side),
    ``timestamp`` (epoch seconds), ``width``/``height`` (positive ints) and
    ``jpegBase64`` (non-empty string, no ``data:`` prefix).
    """
    missing = [
        key for key in ("frameId", "timestamp", "width", "height", "jpegBase64")
        if key not in payload
    ]
    if missing:
        raise ProtocolError(
            "El payload de 'frame.provide' debe incluir frameId, timestamp, "
            f"width, height y jpegBase64; faltan: {', '.join(missing)}."
        )
    frame_id = payload["frameId"]
    if not isinstance(frame_id, int) or isinstance(frame_id, bool):
        raise ProtocolError("'frameId' debe ser un entero.")
    if not _is_number(payload["timestamp"]):
        raise ProtocolError("'timestamp' debe ser un número (epoch en segundos).")
    width = payload["width"]
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ProtocolError("'width' debe ser un entero positivo.")
    height = payload["height"]
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ProtocolError("'height' debe ser un entero positivo.")
    jpeg_base64 = payload["jpegBase64"]
    if not isinstance(jpeg_base64, str) or not jpeg_base64:
        raise ProtocolError("'jpegBase64' debe ser un string no vacío.")
    try:
        base64.b64decode(jpeg_base64, validate=True)
    except (ValueError, TypeError):
        raise ProtocolError("'jpegBase64' debe ser una cadena Base64 válida.") from None


def parse_client_message(raw: str) -> IncomingMessage:
    """Parse and validate one client->server message; raise ``ProtocolError``.

    Validation reflects contracts/messages.ts: the envelope must carry exactly
    ``messageId``/``timestamp``/``protocolVersion``/``kind``/``payload`` with
    correct types, ``protocolVersion == "1"`` and a known client-side kind.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("JSON inválido en el mensaje WebSocket.") from exc
    if not isinstance(data, dict):
        raise ProtocolError("El mensaje WebSocket debe ser un objeto JSON.")
    envelope = cast(dict[str, object], data)
    if set(envelope) != ENVELOPE_KEYS:
        raise ProtocolError(
            "El envelope debe tener exactamente las claves "
            "messageId, timestamp, protocolVersion, kind y payload."
        )

    message_id = envelope["messageId"]
    timestamp = envelope["timestamp"]
    protocol_version = envelope["protocolVersion"]
    kind = envelope["kind"]
    payload = envelope["payload"]

    if not isinstance(message_id, str) or not message_id:
        raise ProtocolError("'messageId' debe ser un string no vacío.")
    if not _is_number(timestamp):
        raise ProtocolError("'timestamp' debe ser un número (epoch en segundos).")
    if protocol_version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"'protocolVersion' debe ser '{PROTOCOL_VERSION}'; recibí "
            f"{protocol_version!r}."
        )
    if not isinstance(kind, str) or kind not in CLIENT_KINDS:
        raise ProtocolError(f"Kind desconocido o no emitible por el cliente: {kind!r}.")
    if not isinstance(payload, dict):
        raise ProtocolError("'payload' debe ser un objeto JSON.")
    if kind == MessageKind.PING and (
        "sentAt" not in payload or not _is_number(payload["sentAt"])
    ):
        raise ProtocolError("El payload de 'ping' debe incluir 'sentAt' numérico.")
    if kind == MessageKind.FRAME_PROVIDE:
        _validate_frame_provide(payload)

    return IncomingMessage(
        message_id=message_id,
        kind=kind,
        payload=cast(dict[str, object], payload),
    )