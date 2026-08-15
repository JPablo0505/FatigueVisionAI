"""Pure OpenCV drawing helpers to overlay detections and meshes on frames.

These are stateless functions: each returns a NEW ``Frame`` with a fresh
copy of the pixel data, leaving the input frame untouched. They are
headless-safe (no GUI, no ``imshow``) and are not invoked by the pipeline;
the orchestrator decides when to draw.
"""

import cv2

from vision_engine.types import FaceDetection, FaceLandmarks, Frame


def draw_detections(frame: Frame, detections: list[FaceDetection]) -> Frame:
    """Return a new frame with green bounding boxes drawn per detection."""
    overlay = frame.data.copy()
    for detection in detections:
        x, y, width, height = detection.bbox
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return Frame(data=overlay, timestamp_ns=frame.timestamp_ns)


def draw_mesh(frame: Frame, meshes: list[FaceLandmarks]) -> Frame:
    """Return a new frame with cyan dots at each normalized landmark point."""
    overlay = frame.data.copy()
    for mesh in meshes:
        for x, y, _z in mesh.points:
            center = (int(x * frame.width), int(y * frame.height))
            cv2.circle(overlay, center, 2, (255, 255, 0), -1)
    return Frame(data=overlay, timestamp_ns=frame.timestamp_ns)