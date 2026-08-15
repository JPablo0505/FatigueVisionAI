"""Exceptions raised by the AI Engine."""


class AIEngineError(Exception):
    """Base error for all AI Engine failures."""


class InvalidFeaturesError(AIEngineError):
    """Raised when fatigue features are out of range or malformed."""


class InferenceError(AIEngineError):
    """Raised when the internal predictor fails during inference."""