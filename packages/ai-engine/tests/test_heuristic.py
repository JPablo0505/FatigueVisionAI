"""Unit tests for the S1-17 HeuristicFatigueModel risk-state classifier.

Covers the threshold ladder (0.30 / 0.50 / 1.00 / 2.00), the corroboration
logic (a single historical long closure must NOT escalate without a second
signal), the active-closure immediate escalation, level/score/confidence
mapping and the degraded-input default (features lacking the new additive
fields classify as normal).
"""

from itertools import pairwise

import pytest

from ai_engine import (
    FatigueFeatures,
    FatigueLevel,
    FatigueResult,
    HeadPose,
    HeuristicFatigueModel,
    IrisDirection,
    RiskState,
)
from ai_engine.heuristic import (
    BLINK_RATE_LOW_CORROBORATION,
    CONFIDENCE_CAP,
    FATIGUE_CLOSURE_MAX_S,
    NORMAL_BLINK_MAX_S,
    PERCLOS_CORROBORATION,
)
from ai_engine.ports import FatigueModel

# A baseline that is NOT corroborated: PERCLOS below the floor, a normal blink
# rate, no yawn evidence.
_CLEAN_PERCLOS = PERCLOS_CORROBORATION - 0.01
_CLEAN_BLINK_RATE = BLINK_RATE_LOW_CORROBORATION + 1.0


def make_features(**overrides: object) -> FatigueFeatures:
    """Return valid FatigueFeatures with the S1-17 fields explicitly set."""
    values: dict[str, object] = {
        "ear": 0.3,
        "mar": 0.2,
        "perclos": _CLEAN_PERCLOS,
        "blink_rate": _CLEAN_BLINK_RATE,
        "head_pose": HeadPose(0.0, 0.0, 0.0),
        "eye_closure_duration": 0.0,
        "yawn_duration": 0.0,
        "iris_direction": IrisDirection(0.0, 0.0),
        "window_seconds": 30.0,
        "current_eye_closure_duration": 0.0,
        "longest_eye_closure_duration": 0.0,
        "longest_yawn_duration": 0.0,
        "total_blinks": 0.0,
        "total_yawns": 0.0,
    }
    values.update(overrides)
    return FatigueFeatures(**values)  # type: ignore[arg-type]


@pytest.fixture()
def model() -> HeuristicFatigueModel:
    return HeuristicFatigueModel()


# -- port conformance ----------------------------------------------------------


def test_heuristic_implements_fatigue_model_port(model: HeuristicFatigueModel) -> None:
    assert isinstance(model, FatigueModel)
    result = model.predict(make_features())
    assert isinstance(result, FatigueResult)
    assert result.timestamp is None  # the pipeline seals it


# -- threshold ladder (historical closures) ------------------------------------


@pytest.mark.parametrize(
    ("longest", "expected"),
    [
        (0.29, RiskState.normal),
        (0.30, RiskState.long_blink),
        (0.49, RiskState.long_blink),
        (0.50, RiskState.long_blink),  # boundary: still a long blink uncorroborated
        (0.99, RiskState.long_blink),
        (1.00, RiskState.fatigue_warning),  # uncorroborated >= 1.0 drops a level
        (1.99, RiskState.fatigue_warning),
        (2.00, RiskState.critical_microsleep),  # unambiguous
        (2.50, RiskState.critical_microsleep),
    ],
)
def test_historical_closure_ladder_boundaries(
    model: HeuristicFatigueModel, longest: float, expected: RiskState
) -> None:
    result = model.predict(make_features(longest_eye_closure_duration=longest))
    assert result.state is expected


def test_historical_closure_boundary_0_30_is_long_blink(
    model: HeuristicFatigueModel,
) -> None:
    just_below = model.predict(
        make_features(longest_eye_closure_duration=NORMAL_BLINK_MAX_S - 0.001)
    )
    exactly = model.predict(
        make_features(longest_eye_closure_duration=NORMAL_BLINK_MAX_S)
    )
    assert just_below.state is RiskState.normal
    assert exactly.state is RiskState.long_blink


def test_historical_closure_boundary_1_00_requires_corroboration(
    model: HeuristicFatigueModel,
) -> None:
    uncorroborated = model.predict(
        make_features(longest_eye_closure_duration=FATIGUE_CLOSURE_MAX_S)
    )
    corroborated = model.predict(
        make_features(
            longest_eye_closure_duration=FATIGUE_CLOSURE_MAX_S,
            perclos=PERCLOS_CORROBORATION,
        )
    )
    assert uncorroborated.state is RiskState.fatigue_warning
    assert corroborated.state is RiskState.microsleep_suspected


# -- corroboration logic -------------------------------------------------------


def test_single_long_closure_without_corroboration_is_fatigue_warning(
    model: HeuristicFatigueModel,
) -> None:
    # A single 1.2 s closure in the history, with NO second signal: the risk
    # must NOT jump to microsleep (false-positive guard).
    result = model.predict(make_features(longest_eye_closure_duration=1.2))
    assert result.state is RiskState.fatigue_warning


def test_long_closure_corroborated_by_perclos_is_microsleep_suspected(
    model: HeuristicFatigueModel,
) -> None:
    result = model.predict(
        make_features(
            longest_eye_closure_duration=1.2,
            perclos=PERCLOS_CORROBORATION + 0.01,
        )
    )
    assert result.state is RiskState.microsleep_suspected


def test_long_closure_corroborated_by_low_blink_rate_is_microsleep_suspected(
    model: HeuristicFatigueModel,
) -> None:
    result = model.predict(
        make_features(
            longest_eye_closure_duration=1.2,
            blink_rate=BLINK_RATE_LOW_CORROBORATION,
        )
    )
    assert result.state is RiskState.microsleep_suspected


def test_long_closure_corroborated_by_completed_yawn_is_microsleep_suspected(
    model: HeuristicFatigueModel,
) -> None:
    result = model.predict(
        make_features(longest_eye_closure_duration=1.2, total_yawns=1.0)
    )
    assert result.state is RiskState.microsleep_suspected


def test_long_closure_corroborated_by_active_yawn_is_microsleep_suspected(
    model: HeuristicFatigueModel,
) -> None:
    result = model.predict(
        make_features(longest_eye_closure_duration=1.2, current_yawn_duration=0.5)
    )
    assert result.state is RiskState.microsleep_suspected


def test_long_blink_corroborated_escalates_to_fatigue_warning(
    model: HeuristicFatigueModel,
) -> None:
    # A 0.6 s closure is a long blink without corroboration but a warning
    # when a second signal backs it up.
    plain = model.predict(make_features(longest_eye_closure_duration=0.6))
    backed = model.predict(
        make_features(longest_eye_closure_duration=0.6, perclos=0.1)
    )
    assert plain.state is RiskState.long_blink
    assert backed.state is RiskState.fatigue_warning


# -- active closure escalates immediately --------------------------------------


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        (0.99, RiskState.normal),  # below the active threshold -> clean history
        (1.00, RiskState.microsleep_suspected),
        (1.50, RiskState.microsleep_suspected),
        (1.99, RiskState.microsleep_suspected),
        (2.00, RiskState.critical_microsleep),
        (2.50, RiskState.critical_microsleep),
    ],
)
def test_active_closure_escalates_immediately_without_corroboration(
    model: HeuristicFatigueModel, current: float, expected: RiskState
) -> None:
    # Active closure needs NO corroboration: the eyes are closed right now.
    result = model.predict(make_features(current_eye_closure_duration=current))
    assert result.state is expected


def test_active_closure_overrides_clean_history(
    model: HeuristicFatigueModel,
) -> None:
    result = model.predict(
        make_features(
            current_eye_closure_duration=1.4,
            longest_eye_closure_duration=0.2,  # clean history
            perclos=0.0,
            blink_rate=20.0,
        )
    )
    assert result.state is RiskState.microsleep_suspected


# -- level mapping -------------------------------------------------------------


@pytest.mark.parametrize(
    ("features", "expected_level"),
    [
        ({"longest_eye_closure_duration": 0.0}, FatigueLevel.low),
        ({"longest_eye_closure_duration": 0.4}, FatigueLevel.low),
        ({"longest_eye_closure_duration": 1.2}, FatigueLevel.medium),
        ({"longest_eye_closure_duration": 1.2, "perclos": 0.1}, FatigueLevel.high),
        ({"current_eye_closure_duration": 1.2}, FatigueLevel.high),
        ({"current_eye_closure_duration": 2.5}, FatigueLevel.high),
    ],
)
def test_level_maps_from_state(
    model: HeuristicFatigueModel,
    features: dict[str, object],
    expected_level: FatigueLevel,
) -> None:
    assert model.predict(make_features(**features)).level is expected_level


def test_state_is_authoritative_over_score_for_level(
    model: HeuristicFatigueModel,
) -> None:
    # fatigue_warning keeps a medium level even though its score (>= 50) would
    # otherwise fall in the default high band.
    warning = model.predict(make_features(longest_eye_closure_duration=1.2))
    assert warning.level is FatigueLevel.medium
    assert warning.state is RiskState.fatigue_warning


# -- score ---------------------------------------------------------------------


def test_score_is_monotonic_with_state(model: HeuristicFatigueModel) -> None:
    scores = [
        model.predict(make_features(longest_eye_closure_duration=0.0)).score,
        model.predict(make_features(longest_eye_closure_duration=0.4)).score,
        model.predict(make_features(longest_eye_closure_duration=1.2)).score,
        model.predict(
            make_features(longest_eye_closure_duration=1.2, perclos=0.1)
        ).score,
        model.predict(make_features(current_eye_closure_duration=2.5)).score,
    ]
    assert all(later > earlier for earlier, later in pairwise(scores))


def test_corroboration_nudges_score_up_within_the_band(
    model: HeuristicFatigueModel,
) -> None:
    plain = model.predict(make_features(longest_eye_closure_duration=1.2))
    backed = model.predict(
        make_features(longest_eye_closure_duration=1.2, perclos=0.1)
    )
    assert plain.state is RiskState.fatigue_warning
    assert backed.state is RiskState.microsleep_suspected
    assert plain.score < 70.0
    assert 70.0 <= backed.score < 88.0
    assert 0.0 <= plain.score <= 100.0
    assert 0.0 <= backed.score <= 100.0


def test_yawn_evidence_nudges_score_up_within_the_band(
    model: HeuristicFatigueModel,
) -> None:
    plain = model.predict(make_features(longest_eye_closure_duration=1.2))
    yawning = model.predict(
        make_features(longest_eye_closure_duration=1.2, total_yawns=1.0)
    )
    assert yawning.score > plain.score
    assert yawning.state is RiskState.microsleep_suspected


# -- confidence ----------------------------------------------------------------


def test_confidence_baseline_without_corroboration(
    model: HeuristicFatigueModel,
) -> None:
    result = model.predict(make_features(longest_eye_closure_duration=0.4))
    assert result.confidence == pytest.approx(0.5)


def test_confidence_bumped_by_corroboration(model: HeuristicFatigueModel) -> None:
    backed = model.predict(
        make_features(longest_eye_closure_duration=1.2, perclos=0.1)
    )
    assert backed.confidence == pytest.approx(0.7)


def test_confidence_bumped_by_yawn_evidence(model: HeuristicFatigueModel) -> None:
    # Yawn evidence is itself a corroboration signal, so it contributes BOTH
    # the corroboration bonus and the yawn bonus: 0.5 + 0.2 + 0.15.
    yawning = model.predict(make_features(longest_eye_closure_duration=1.2, total_yawns=1.0))
    assert yawning.confidence == pytest.approx(0.85)


def test_confidence_never_exceeds_the_cap(model: HeuristicFatigueModel) -> None:
    # Every bonus stacked at once still cannot breach the defensive cap.
    fully_backed = model.predict(
        make_features(
            longest_eye_closure_duration=1.2,
            perclos=0.1,
            blink_rate=6.0,
            total_yawns=1.0,
        )
    )
    assert fully_backed.confidence == pytest.approx(0.85)
    assert fully_backed.confidence <= CONFIDENCE_CAP


# -- degraded input (additive fields absent) ------------------------------------


def test_default_features_classify_as_normal(model: HeuristicFatigueModel) -> None:
    # An old feature dict / plain constructor lacks every S1-17 field: the
    # defaults classify as a clean normal state with a low score.
    features = FatigueFeatures(
        ear=0.3,
        mar=0.2,
        perclos=0.0,
        blink_rate=15.0,
        head_pose=HeadPose(0.0, 0.0, 0.0),
        eye_closure_duration=0.4,
        yawn_duration=0.0,
        iris_direction=IrisDirection(0.0, 0.0),
        window_seconds=30.0,
    )
    result = model.predict(features)
    assert result.state is RiskState.normal
    assert result.level is FatigueLevel.low
    assert 0.0 <= result.score < 30.0


def test_to_dict_serializes_state(model: HeuristicFatigueModel) -> None:
    result = model.predict(make_features(current_eye_closure_duration=2.5))
    payload = result.to_dict()
    assert payload["state"] == "critical_microsleep"
    assert payload["level"] == "high"


def test_pipeline_preserves_state_when_sealing_timestamp() -> None:
    from ai_engine import FatiguePipeline

    pipeline = FatiguePipeline(HeuristicFatigueModel())
    result = pipeline.assess(make_features(current_eye_closure_duration=1.4))
    assert result.timestamp is not None
    assert result.state is RiskState.microsleep_suspected
    assert result.to_dict()["state"] == "microsleep_suspected"