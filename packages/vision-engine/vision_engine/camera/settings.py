"""Camera configuration settings, validated at construction time."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraSettings:
    """Capture settings passed programmatically to a camera source.

    Values are validated in ``__post_init__``; the package reads all
    configuration through this object and never from the environment.
    """

    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.device_index < 0:
            raise ValueError(
                f"device_index must be >= 0, got {self.device_index}"
            )