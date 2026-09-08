"""Realtime WebSocket endpoint: streaming protocol on ``/ws`` (S1-15).

The route is deliberately thin: it performs the handshake, registers the
socket with the StreamingManager (which owns the actual capture loop) and
bridges the socket's receive loop with the manager's broadcast queues.
Sending and receiving run concurrently: the manager writes into the
subscriber's outbox and a dedicated sender task drains it into the socket,
so a slow client never blocks the camera loop.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import traceback
from contextlib import suppress
from typing import cast

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from vision_engine.types import Frame

from app.protocol import (
    ERROR_CAMERA_UNAVAILABLE,
    ERROR_PROTOCOL,
    WS_PATH,
    MessageKind,
    ProtocolError,
    build_envelope,
    parse_client_message,
)
from app.services.streaming import StreamingManager, Subscriber

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_manager(websocket: WebSocket) -> StreamingManager:
    app = cast(FastAPI, websocket.scope["app"])
    return cast(StreamingManager, app.state.streaming_manager)


def _submit_frame_provide(
    manager: StreamingManager, subscriber: Subscriber, payload: dict[str, object]
) -> None:
    """Decode a browser frame and feed it to the streaming loop.

    Decodes the JPEG payload, builds a ``Frame`` and submits it through the
    manager. A frame that fails to decode (``bgr is None``) is treated as a
    failed frame: the connection survives and the subscriber gets one error
    envelope; the submission result is otherwise ignored (drop semantics).
    """
    try:
        jpeg = base64.b64decode(cast(str, payload["jpegBase64"]))
    except (ValueError, TypeError):
        bgr = None
    else:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        logger.warning("Frame decode failed (jpegBase64 invalid)")
        manager.deliver(
            subscriber,
            build_envelope(
                MessageKind.ERROR,
                {
                    "code": ERROR_CAMERA_UNAVAILABLE,
                    "message": "No se pudo decodificar el frame enviado por el navegador.",
                },
            ),
        )
        return
    ok = manager.submit_frame(Frame(data=bgr, timestamp_ns=time.time_ns()))
    if not ok:
        logger.warning("Frame submitted but dropped by camera source")


async def _sender_loop(websocket: WebSocket, subscriber: Subscriber) -> None:
    """Drain the subscriber's outbox into the socket until the socket breaks."""
    try:
        while True:
            payload = await subscriber.queue.get()
            await websocket.send_text(payload)
    except Exception:  # noqa: BLE001 - broken socket: flag it dead and stop
        logger.debug("Sender loop exit (socket broken or cancelled)")
        subscriber.mark_dead()
        return


@router.websocket(WS_PATH)
async def ws_endpoint(websocket: WebSocket) -> None:
    """Accept one socket, stream contract envelopes, handle control messages."""
    manager = _get_manager(websocket)
    await websocket.accept()
    subscriber = Subscriber()
    await manager.subscribe(subscriber)
    send_task = asyncio.create_task(_sender_loop(websocket, subscriber))
    try:
        while True:
            try:
                message = parse_client_message(await websocket.receive_text())
            except ProtocolError as exc:
                manager.deliver(
                    subscriber,
                    build_envelope(
                        MessageKind.ERROR,
                        {"code": ERROR_PROTOCOL, "message": str(exc)},
                    ),
                )
                continue
            if message.kind == MessageKind.FRAME_PROVIDE:
                _submit_frame_provide(manager, subscriber, message.payload)
                continue
            await manager.handle_client_message(subscriber, message)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception:  # noqa: BLE001 — unexpected error: close cleanly (S3-02)
        logger.error("Unhandled exception in WS endpoint:\n%s", traceback.format_exc())
        with suppress(Exception):
            await websocket.close(code=1011, reason="internal error")
    finally:
        send_task.cancel()
        with suppress(asyncio.CancelledError):
            await send_task
        await manager.unsubscribe(subscriber)