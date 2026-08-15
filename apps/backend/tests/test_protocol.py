"""Unit tests for ``app.protocol.parse_client_message``.

Covers the envelope contract AND the ``frame.provide`` payload shape:
field presence, per-field types/ranges and that the kind belongs to the
client->server vocabulary. The raw bytes of a well-formed-but-unrelated
Base64 string still parse (a failed JPEG is handled at the route, where it
becomes a ``camera.unavailable`` error, not a protocol error).
"""

from __future__ import annotations

import base64
import json
import time

import cv2
import numpy as np
import pytest

from app.protocol import CLIENT_KINDS, MessageKind, ProtocolError, parse_client_message


def _envelope(kind: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "messageId": "m-1",
        "timestamp": time.time(),
        "protocolVersion": "1",
        "kind": kind,
        "payload": payload,
    }


def _jpeg_b64() -> str:
    ok, buf = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _frame_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "frameId": 1,
        "timestamp": time.time(),
        "width": 16,
        "height": 16,
        "jpegBase64": _jpeg_b64(),
    }
    payload.update(overrides)
    return payload


def _parse(payload: dict[str, object]):
    return parse_client_message(json.dumps(_envelope(MessageKind.FRAME_PROVIDE, payload)))


def test_frame_provide_is_in_the_client_vocabulary() -> None:
    assert MessageKind.FRAME_PROVIDE == "frame.provide"
    assert MessageKind.FRAME_PROVIDE in CLIENT_KINDS


def test_valid_frame_provide_parses() -> None:
    payload = _frame_payload()
    message = _parse(payload)
    assert message.kind == MessageKind.FRAME_PROVIDE
    assert message.payload == payload
    assert message.message_id == "m-1"


@pytest.mark.parametrize(
    "missing", ["frameId", "timestamp", "width", "height", "jpegBase64"]
)
def test_frame_provide_missing_field_is_protocol_error(missing: str) -> None:
    payload = _frame_payload()
    del payload[missing]
    with pytest.raises(ProtocolError, match="debe incluir frameId, timestamp"):
        _parse(payload)


def test_frame_provide_zero_width_is_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="'width' debe ser un entero positivo"):
        _parse(_frame_payload(width=0))


def test_frame_provide_negative_height_is_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="'height' debe ser un entero positivo"):
        _parse(_frame_payload(height=-1))


def test_frame_provide_float_frame_id_is_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="'frameId' debe ser un entero"):
        _parse(_frame_payload(frameId=1.0))


def test_frame_provide_bool_frame_id_is_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="'frameId' debe ser un entero"):
        _parse(_frame_payload(frameId=True))


def test_frame_provide_string_width_is_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="'width' debe ser un entero positivo"):
        _parse(_frame_payload(width="16"))


def test_frame_provide_empty_jpeg_is_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="'jpegBase64' debe ser un string no vacío"):
        _parse(_frame_payload(jpegBase64=""))


def test_frame_provide_non_string_jpeg_is_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="'jpegBase64' debe ser un string no vacío"):
        _parse(_frame_payload(jpegBase64=123))


def test_frame_provide_bad_base64_is_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="Base64 válida"):
        _parse(_frame_payload(jpegBase64="###not-base64####"))


def test_frame_provide_valid_base64_non_jpeg_parses_at_protocol_level() -> None:
    # The protocol only checks the shape; whether the bytes decode to an image
    # belongs to the route, where a non-image becomes a failed frame.
    payload = _frame_payload(jpegBase64=base64.b64encode(b"not a jpeg").decode("ascii"))
    message = _parse(payload)
    assert message.kind == MessageKind.FRAME_PROVIDE
    assert message.payload["jpegBase64"] == payload["jpegBase64"]


def test_ping_payload_still_validated() -> None:
    message = parse_client_message(
        json.dumps(_envelope(MessageKind.PING, {"sentAt": time.time()}))
    )
    assert message.kind == MessageKind.PING
    with pytest.raises(ProtocolError, match="'sentAt'"):
        parse_client_message(json.dumps(_envelope(MessageKind.PING, {})))