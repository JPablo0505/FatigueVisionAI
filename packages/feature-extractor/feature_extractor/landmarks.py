"""Reference landmark indices for the MediaPipe FaceMesh 468-point model.

This module centralizes the face-landmark indices the feature extractor
reads from a ``FaceLandmarks`` mesh. MediaPipe FaceMesh emits exactly 468
points in normalized image coordinates; the indices below follow the
canonical convention used across the FaceMesh reference cards and the
eye-blink / yawn literature.

Side convention (subject vs. image): MediaPipe FaceMesh describes the face
as seen in a mirrored (selfie) view, so the group labeled ``LEFT_EYE`` is
the eye that appears on the viewer's left side of the image. When the input
is not mirrored the labels swap relative to the anatomical subject. The
extractor averages both eyes and computes pose from both-eye geometry, so
the aggregate ``FatigueFeatures`` are robust to this convention choice.

These are reference constants and may be replaced if a different landmark
model or a recalibrated index set is adopted in a future task; nothing in
the extractor is allowed to hardcode indices outside this module.

References:
    - MediaPipe FaceMesh annotations: 468 face landmarks.
    - EAR formulation: Soukupová & Čech, "Real-Time Eye Blink Detection
      using Facial Landmarks" (2016), adapted to the eye index pairs
      ``(33, 160, 158, 133, 153, 144)`` and ``(362, 385, 387, 263, 373, 380)``.
    - MAR formulation: vertical lip inner points (13, 14) over the mouth
      corner width (61, 291).
    - Iris ring points: right iris 468-472, left iris 473-477.
"""

from typing import Final

LEFT_EYE: Final[tuple[int, int, int, int, int, int]] = (33, 160, 158, 133, 153, 144)
"""Left eye (viewer side), canonical six-point ordering.

Ordering is ``(outer_corner, upper_1, lower_2, inner_corner, upper_2,
lower_1)`` so that ``EAR = (dist(p[1], p[5]) + dist(p[2], p[4])) /
(2 * dist(p[0], p[3]))``.
"""

RIGHT_EYE: Final[tuple[int, int, int, int, int, int]] = (362, 385, 387, 263, 373, 380)
"""Right eye (viewer side), same six-point ordering as ``LEFT_EYE``."""

MOUTH_INNER_LIP: Final[tuple[int, int]] = (13, 14)
"""Inner lip landmarks measured vertically to estimate mouth opening."""

MOUTH_CORNERS: Final[tuple[int, int]] = (61, 291)
"""Outer mouth-corner landmarks; their distance normalizes the mouth width."""

RIGHT_IRIS: Final[tuple[int, int, int, int, int]] = (468, 469, 470, 471, 472)
"""Right iris ring points (viewer side), centroid approximates the iris center."""

LEFT_IRIS: Final[tuple[int, int, int, int, int]] = (473, 474, 475, 476, 477)
"""Left iris ring points (viewer side), centroid approximates the iris center."""

NOSE_TIP: Final[int] = 4
"""Nose tip landmark used by the approximate head-pose heuristic."""

__all__ = [
    "LEFT_EYE",
    "LEFT_IRIS",
    "MOUTH_CORNERS",
    "MOUTH_INNER_LIP",
    "NOSE_TIP",
    "RIGHT_EYE",
    "RIGHT_IRIS",
]