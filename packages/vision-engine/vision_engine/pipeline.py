"""Composition root for deterministic, synchronous frame processing."""

from vision_engine.ports import FaceDetector, FaceMeshEstimator, FrameProcessor
from vision_engine.types import Frame, VisionEvent


class VisionPipeline:
    """Pipelines optional vision stages into a single processing unit.

    All stages are injected through the constructor so the pipeline works
    in isolation with fakes and later with real adapters (composition over
    inheritance). Purely synchronous; performs no I/O by itself.
    """

    def __init__(
        self,
        detector: FaceDetector | None = None,
        mesh: FaceMeshEstimator | None = None,
        processor: FrameProcessor | None = None,
        source: str = "camera",
    ) -> None:
        """Build a pipeline from optional stage implementations."""
        self._detector = detector
        self._mesh = mesh
        self._processor = processor
        self._source = source
        self._frame_id = 0

    def process_frame(self, frame: Frame) -> VisionEvent:
        """Run the configured stages over the frame and return the event.

        A processor, if set, is applied before detection and mesh
        estimation. Every call produces an event with an incrementing
        ``frame_id`` starting at 1.
        """
        processed = frame if self._processor is None else self._processor.process(frame)
        detections = (
            [] if self._detector is None else self._detector.detect(processed)
        )
        meshes = (
            [] if self._mesh is None else self._mesh.estimate(processed)
        )
        self._frame_id += 1
        return VisionEvent(
            frame_id=self._frame_id,
            timestamp_ns=frame.timestamp_ns,
            source=self._source,
            detections=detections,
            meshes=meshes,
        )