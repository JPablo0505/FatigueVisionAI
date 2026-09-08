"""Vision Engine adapter: one-shot status and capture over camera + pipeline.

Engine resources are built lazily on first use so a missing MediaPipe model
reports a clean pipeline error instead of crashing the process at startup.
"""

from __future__ import annotations

import logging
from typing import cast

from vision_engine.camera import CameraSettings, create_camera_source
from vision_engine.errors import ProcessingError
from vision_engine.mediapipe import MediaPipeFaceDetector, MediaPipeFaceMesh
from vision_engine.pipeline import VisionPipeline
from vision_engine.ports import CameraSource
from vision_engine.types import Frame, VisionEvent

from app.adapters.browser_camera import BrowserFrameSource
from app.config import Settings

logger = logging.getLogger(__name__)


class VisionAdapter:
    """Facade over the Vision Engine for the ``VisionStatus``/``VisionEvent`` API."""

    def __init__(
        self,
        settings: Settings,
        *,
        camera: CameraSource | None = None,
        pipeline: VisionPipeline | None = None,
    ) -> None:
        self._settings = settings
        # The injected objects are PROTOTYPES: they are reused on every rebuild
        # (tests inject fakes here). Only when no prototype exists does the
        # adapter build the real camera/pipeline from settings.
        self._camera_proto = camera
        self._pipeline_proto = pipeline
        self._camera = camera
        self._pipeline = pipeline
        self._build_error: str | None = None

    def _ensure_resources(self) -> None:
        """Build camera and pipeline once; cache any construction error."""
        if self._build_error is not None:
            return
        if self._camera is not None and self._pipeline is not None:
            return
        try:
            if self._camera is None:
                if self._camera_proto is not None:
                    self._camera = self._camera_proto
                elif self._settings.camera_source == "browser":
                    # Browser mode: no device. Frames arrive over the WebSocket
                    # via ``frame.provide`` and are submitted to this source.
                    self._camera = BrowserFrameSource()
                else:
                    camera_settings = CameraSettings(
                        device_index=self._settings.camera_index,
                        width=self._settings.synthetic_width,
                        height=self._settings.synthetic_height,
                        fps=self._settings.synthetic_fps,
                    )
                    self._camera = create_camera_source(
                        self._settings.camera_source, camera_settings
                    )
            if self._pipeline is None:
                if self._pipeline_proto is not None:
                    self._pipeline = self._pipeline_proto
                else:
                    detector = MediaPipeFaceDetector(str(self._settings.face_detector_model))
                    mesh = MediaPipeFaceMesh(str(self._settings.face_landmarker_model))
                    self._pipeline = VisionPipeline(
                        detector=detector, mesh=mesh, source=self._settings.camera_source
                    )
        except ProcessingError as exc:
            self._build_error = str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaces any engine startup failure
            self._build_error = str(exc)

    def status(self) -> dict[str, object]:
        """Return the ``VisionStatus``-shaped dict for the configured camera.

        ``resolution``/``fps`` are the values the camera was configured with
        (the ``CameraSource`` port exposes no live introspection). The
        pipeline is ``stopped`` until a streaming loop exists (S1-15).
        """
        source = self._settings.camera_source
        if self._build_error is not None:
            return {
                "camera": {
                    "source": source,
                    "running": False,
                    "resolution": None,
                    "fps": None,
                },
                "pipeline": "error",
                "error": self._build_error,
            }
        running = self._camera is not None and not self._camera.closed
        return {
            "camera": {
                "source": source,
                "running": running,
                "resolution": {
                    "width": self._settings.synthetic_width,
                    "height": self._settings.synthetic_height,
                },
                "fps": float(self._settings.synthetic_fps),
            },
            "pipeline": "stopped",
            "error": None,
        }

    def _capture_event(self) -> VisionEvent:
        """Read one frame and run it through the pipeline, returning the event."""
        self._ensure_resources()
        if self._build_error is not None:
            raise ProcessingError(self._build_error)
        assert self._camera is not None
        assert self._pipeline is not None
        frame = self._camera.read()
        return self._pipeline.process_frame(frame)

    def capture_event(self) -> VisionEvent:
        """Read one frame and return the raw ``VisionEvent``.

        Downstream consumers (the ``FeatureExtractor``) need the landmark
        object, not its JSON projection, so this adapter exposes the event
        itself in addition to the dict form used by ``/api/vision/frame``.
        """
        return cast(VisionEvent, self._capture_event())

    def capture_once(self) -> dict[str, object]:
        """Read one frame and return the processed ``VisionEvent`` as a dict.

        One-off capture mode until the realtime loop arrives with S1-15.
        """
        return cast(dict[str, object], self._capture_event().to_dict())

    def submit_frame(self, frame: Frame) -> bool:
        """Feed one externally captured frame into the camera source.

        Only meaningful in browser mode: the WebSocket route decodes a
        ``frame.provide`` payload into a ``Frame`` and pushes it here so the
        streaming loop's ``capture_event()`` picks it up from the queue.
        Returns False when this adapter has no frame source that accepts
        submissions (drop semantics, e.g. the source is closed).
        """
        self._ensure_resources()
        if self._build_error is not None or self._camera is None:
            logger.warning("submit_frame dropped: build_error=%s", self._build_error)
            return False
        submitter = getattr(self._camera, "submit", None)
        if submitter is None:
            logger.warning(
                "submit_frame dropped: camera source %s has no submit() — "
                "is CAMERA_SOURCE=browser?",
                type(self._camera).__name__,
            )
            return False
        return cast(bool, submitter(frame))

    def close(self) -> None:
        """Release the camera and the pipeline stages; safe to call repeatedly.

        The engine stages expose an optional ``close()`` (MediaPipe tasks spin
        up native workers, so leaving them open leaks threads). Fakes without
        a ``close`` are skipped.

        Resources are also dropped (set back to ``None``) so a later
        ``start()`` rebuilds the camera and pipeline from scratch — this is
        what lets the streaming manager release the real device on ``stop``
        and reacquire it on the next run.
        """
        if self._camera is not None and not self._camera.closed:
            self._camera.close()
        if self._pipeline is not None:
            for name in ("_detector", "_mesh"):
                stage = getattr(self._pipeline, name, None)
                closer = getattr(stage, "close", None)
                if closer is not None:
                    closer()
        self._camera = None
        self._pipeline = None
        self._build_error = None