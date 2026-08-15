"""Exceptions raised by the Vision Engine."""


class VisionEngineError(Exception):
    """Base error for all Vision Engine failures."""


class CameraNotAvailableError(VisionEngineError):
    """Raised when the camera source cannot be opened or read."""


class ProcessingError(VisionEngineError):
    """Raised when a frame cannot be processed or its payload is invalid."""