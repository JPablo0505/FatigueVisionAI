"""Unit tests for the FeatureExtractor public API."""

import json

import pytest
from ai_engine import FatigueFeatures
from vision_engine import FaceDetection, VisionEvent

from feature_extractor import FeatureExtractor, FeatureExtractorConfig
from tests.helpers import build_face


def _event(
    ts_seconds: float,
    detections: list,
    meshes: list,
) -> VisionEvent:
    """Build a synthetic VisionEvent for a given timestamp (seconds)."""
    return VisionEvent(
        frame_id=1,
        timestamp_ns=int(ts_seconds * 1_000_000_000),
        source="synthetic",
        detections=detections,
        meshes=meshes,
    )


def _one_detection() -> list:
    """A single large detection paired by index with the first mesh."""
    return [FaceDetection(bbox=(0, 0, 200, 200), score=0.95)]


def test_update_no_usable_face_returns_none() -> None:
    extractor = FeatureExtractor(FeatureExtractorConfig(window_seconds=30.0))
    assert extractor.update(_event(0.0, [], [])) is None
    no_mesh = _event(
        1.0 / 30.0, [FaceDetection(bbox=(0, 0, 10, 10), score=0.5)], []
    )
    assert extractor.update(no_mesh) is None

    open_face = build_face(left_ear=0.5, right_ear=0.5)
    features = extractor.update(_event(2.0 / 30.0, _one_detection(), [open_face]))
    assert features is not None
    assert features.ear == pytest.approx(0.5, abs=1e-9)


def test_update_streams_valid_features_and_json_dumps() -> None:
    extractor = FeatureExtractor(FeatureExtractorConfig(window_seconds=30.0))
    face = build_face(left_ear=0.5, right_ear=0.5)
    first = extractor.update(_event(0.0, _one_detection(), [face]))
    second = extractor.update(_event(1.0 / 30.0, _one_detection(), [face]))
    assert first is not None
    assert second is not None
    assert isinstance(second, FatigueFeatures)

    payload = second.to_dict()
    json.dumps(payload)
    assert set(payload) == {
        "ear",
        "mar",
        "perclos",
        "blink_rate",
        "head_pose",
        "eye_closure_duration",
        "yawn_duration",
        "iris_direction",
        "window_seconds",
        "current_yawn_duration",
        "last_yawn_duration",
        "mar_instant",
        "ear_instant",
        "current_eye_closure_duration",
        "longest_eye_closure_duration",
        "longest_yawn_duration",
        "total_blinks",
        "total_yawns",
    }


def test_largest_face_selected() -> None:
    extractor = FeatureExtractor(FeatureExtractorConfig(window_seconds=30.0))
    small = FaceDetection(bbox=(0, 0, 50, 50), score=0.6)
    big = FaceDetection(bbox=(0, 0, 200, 200), score=0.95)
    open_face = build_face(left_ear=0.5, right_ear=0.5)
    closed_face = build_face(left_ear=0.0, right_ear=0.0)

    # The open face pairs with the BIG detection even though it comes second.
    features = extractor.update(_event(0.0, [small, big], [closed_face, open_face]))
    assert features is not None
    assert features.ear == pytest.approx(0.5, abs=1e-9)


def test_window_seconds_exposed() -> None:
    assert FeatureExtractor().window_seconds == 5.0
    custom = FeatureExtractor(FeatureExtractorConfig(window_seconds=15.0))
    assert custom.window_seconds == 15.0


def test_reset_clears_state() -> None:
    extractor = FeatureExtractor(FeatureExtractorConfig(window_seconds=30.0))
    closed_face = build_face(left_ear=0.0, right_ear=0.0)
    open_face = build_face(left_ear=0.5, right_ear=0.5)

    extractor.update(_event(0.0, _one_detection(), [closed_face]))
    before = extractor.update(_event(1.0 / 30.0, _one_detection(), [open_face]))
    assert before is not None
    assert before.ear == pytest.approx(0.25, abs=1e-9)

    extractor.reset()
    after = extractor.update(_event(2.0 / 30.0, _one_detection(), [open_face]))
    assert after is not None
    assert after.ear == pytest.approx(0.5, abs=1e-9)