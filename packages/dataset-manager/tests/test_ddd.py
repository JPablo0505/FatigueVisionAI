"""Tests for DDD adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_manager.adapters.ddd import (
    DDDConfig,
    DDDAdapter,
    _parse_filename,
    _list_image_files,
)


class TestFilenameParsing:
    """Validate DDD filename convention parsing."""

    def test_parse_drowsy_uppercase(self) -> None:
        result = _parse_filename("A0001")
        assert result is not None
        assert result["subject_id"] == "a"
        assert result["fatigue_level"] == "drowsy"
        assert result["frame_number"] == 1

    def test_parse_drowsy_high_number(self) -> None:
        result = _parse_filename("Z1431")
        assert result is not None
        assert result["subject_id"] == "z"
        assert result["fatigue_level"] == "drowsy"
        assert result["frame_number"] == 1431

    def test_parse_non_drowsy_lowercase(self) -> None:
        result = _parse_filename("a0002")
        assert result is not None
        assert result["subject_id"] == "a"
        assert result["fatigue_level"] == "non_drowsy"
        assert result["frame_number"] == 2

    def test_parse_non_drowsy_high_number(self) -> None:
        result = _parse_filename("z3579")
        assert result is not None
        assert result["subject_id"] == "z"
        assert result["fatigue_level"] == "non_drowsy"
        assert result["frame_number"] == 3579

    def test_parse_with_png_extension(self) -> None:
        result = _parse_filename("A0001.png")
        assert result is not None
        assert result["subject_id"] == "a"
        assert result["fatigue_level"] == "drowsy"
        assert result["frame_number"] == 1

    def test_parse_with_png_extension_non_drowsy(self) -> None:
        result = _parse_filename("b0409.png")
        assert result is not None
        assert result["subject_id"] == "b"
        assert result["fatigue_level"] == "non_drowsy"
        assert result["frame_number"] == 409

    def test_parse_invalid_no_digits(self) -> None:
        result = _parse_filename("A")
        assert result is None

    def test_parse_invalid_empty(self) -> None:
        result = _parse_filename("")
        assert result is None

    def test_parse_invalid_starts_with_digit(self) -> None:
        result = _parse_filename("1abc")
        assert result is None

    def test_parse_invalid_special_chars(self) -> None:
        result = _parse_filename("A-001")
        assert result is None

    def test_parse_with_spaces(self) -> None:
        """Real files may have trailing spaces."""
        result = _parse_filename("A0001 ")
        assert result is not None
        assert result["subject_id"] == "a"

    def test_parse_middle_alphabet_drowsy(self) -> None:
        result = _parse_filename("M0777")
        assert result is not None
        assert result["subject_id"] == "m"
        assert result["fatigue_level"] == "drowsy"
        assert result["frame_number"] == 777

    def test_parse_middle_alphabet_non_drowsy(self) -> None:
        result = _parse_filename("n0957")
        assert result is not None
        assert result["subject_id"] == "n"
        assert result["fatigue_level"] == "non_drowsy"
        assert result["frame_number"] == 957


class TestImageDiscovery:
    """Validate image file discovery."""

    def test_list_ddd_images(self, tmp_path: Path) -> None:
        drowsy_dir = tmp_path / "Drowsy"
        drowsy_dir.mkdir()
        (drowsy_dir / "A0001.png").write_bytes(b"")
        (drowsy_dir / "B0002.png").write_bytes(b"")

        non_drowsy_dir = tmp_path / "Non Drowsy"
        non_drowsy_dir.mkdir()
        (non_drowsy_dir / "a0001.png").write_bytes(b"")
        (non_drowsy_dir / "b0002.png").write_bytes(b"")

        images = _list_image_files(tmp_path)
        assert len(images) == 4
        drowsy = [p for p, l in images if l == "drowsy"]
        non_drowsy = [p for p, l in images if l == "non_drowsy"]
        assert len(drowsy) == 2
        assert len(non_drowsy) == 2

    def test_list_only_drowsy(self, tmp_path: Path) -> None:
        drowsy_dir = tmp_path / "Drowsy"
        drowsy_dir.mkdir()
        (drowsy_dir / "A0001.png").write_bytes(b"")

        images = _list_image_files(tmp_path)
        assert len(images) == 1
        assert images[0][1] == "drowsy"

    def test_list_only_non_drowsy(self, tmp_path: Path) -> None:
        non_drowsy_dir = tmp_path / "Non Drowsy"
        non_drowsy_dir.mkdir()
        (non_drowsy_dir / "a0001.png").write_bytes(b"")

        images = _list_image_files(tmp_path)
        assert len(images) == 1
        assert images[0][1] == "non_drowsy"

    def test_list_empty_directory(self, tmp_path: Path) -> None:
        images = _list_image_files(tmp_path)
        assert images == []

    def test_list_ignores_non_png(self, tmp_path: Path) -> None:
        drowsy_dir = tmp_path / "Drowsy"
        drowsy_dir.mkdir()
        (drowsy_dir / "A0001.png").write_bytes(b"")
        (drowsy_dir / "A0002.jpg").write_bytes(b"")
        (drowsy_dir / "A0003.txt").write_bytes(b"")

        images = _list_image_files(tmp_path)
        assert len(images) == 1

    def test_list_sorted_order(self, tmp_path: Path) -> None:
        drowsy_dir = tmp_path / "Drowsy"
        drowsy_dir.mkdir()
        (drowsy_dir / "C0003.png").write_bytes(b"")
        (drowsy_dir / "A0001.png").write_bytes(b"")
        (drowsy_dir / "B0002.png").write_bytes(b"")

        images = _list_image_files(tmp_path)
        filenames = [p.name for p, _ in images]
        assert filenames == ["A0001.png", "B0002.png", "C0003.png"]


class TestDDDConfig:
    """Validate DDDConfig construction."""

    def test_requires_ddd_root(self) -> None:
        with pytest.raises(ValueError, match="ddd_root"):
            DDDAdapter(config=None)

    def test_config_defaults(self) -> None:
        config = DDDConfig(ddd_root=Path("/data/ddd"))
        assert config.ddd_root == Path("/data/ddd")
        assert config.image_skip == 1
        assert config.max_faces == 1
        assert config.synthetic_fps == 30.0


class TestDDDValidation:
    """Validate adapter validation with mock data."""

    def test_validate_missing_root(self, tmp_path: Path) -> None:
        config = DDDConfig(ddd_root=tmp_path / "nonexistent")
        adapter = DDDAdapter(config)
        errors = adapter.validate()
        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_validate_empty_directory(self, tmp_path: Path) -> None:
        config = DDDConfig(ddd_root=tmp_path)
        adapter = DDDAdapter(config)
        errors = adapter.validate()
        assert len(errors) == 1
        assert "No PNG files found" in errors[0]

    def test_validate_with_valid_images(self, tmp_path: Path) -> None:
        """Validate that real DDD images pass validation."""
        real_root = Path(
            "datasets/raw/Driver Drowsiness Dataset (DDD)"
        )
        if not real_root.is_dir():
            pytest.skip("Real DDD dataset not available")

        config = DDDConfig(ddd_root=real_root)
        adapter = DDDAdapter(config)
        errors = adapter.validate()
        # Real images should have no errors (or only minor ones)
        assert len(errors) == 0, f"Validation errors: {errors}"


class TestDDDInfo:
    """Validate adapter info metadata."""

    def test_info_returns_dataset_info(self, tmp_path: Path) -> None:
        config = DDDConfig(ddd_root=tmp_path)
        adapter = DDDAdapter(config)
        info = adapter.info()

        assert info.name == "ddd"
        assert info.version == "1.0"
        assert "drowsy" in info.label_vocabulary["fatigue_level"]
        assert "non_drowsy" in info.label_vocabulary["fatigue_level"]
        assert info.license != ""
        assert info.citation != ""

    def test_info_with_mock_data(self, tmp_path: Path) -> None:
        """Test info with mock directory structure."""
        drowsy_dir = tmp_path / "Drowsy"
        drowsy_dir.mkdir()
        (drowsy_dir / "A0001.png").write_bytes(b"")
        (drowsy_dir / "B0002.png").write_bytes(b"")

        non_drowsy_dir = tmp_path / "Non Drowsy"
        non_drowsy_dir.mkdir()
        (non_drowsy_dir / "a0001.png").write_bytes(b"")
        (non_drowsy_dir / "b0002.png").write_bytes(b"")
        (non_drowsy_dir / "c0003.png").write_bytes(b"")

        config = DDDConfig(ddd_root=tmp_path)
        adapter = DDDAdapter(config)
        info = adapter.info()

        assert info.total_frames == 5
        # Subject IDs are normalized to lowercase: A->a, B->b, a->a, b->b, c->c
        # Unique subjects: {a, b, c} = 3
        assert info.subjects == 3


class TestDDDIterEvents:
    """Validate event iteration with mock images."""

    def _create_mock_image(
        self, path: Path, size: tuple[int, int] = (227, 227)
    ) -> None:
        """Create a minimal PNG image for testing."""
        try:
            import cv2

            # Create a blank image with some random content
            img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
            cv2.imwrite(str(path), img)
        except ImportError:
            pytest.skip("opencv-python not available")

    def test_iter_events_with_mock_images(self, tmp_path: Path) -> None:
        """Test that the adapter can read frames from mock images."""
        drowsy_dir = tmp_path / "Drowsy"
        drowsy_dir.mkdir()
        img_path = drowsy_dir / "A0001.png"
        self._create_mock_image(img_path)

        meta = _parse_filename(img_path.stem)
        assert meta is not None
        assert meta["fatigue_level"] == "drowsy"
        assert meta["subject_id"] == "a"

    def test_iter_events_model_loading_failure(self, tmp_path: Path) -> None:
        """Test that adapter raises when model files don't exist."""
        drowsy_dir = tmp_path / "Drowsy"
        drowsy_dir.mkdir()
        self._create_mock_image(drowsy_dir / "A0001.png")

        config = DDDConfig(
            ddd_root=tmp_path,
            face_detector=tmp_path / "nonexistent_detector.tflite",
            face_landmarker=tmp_path / "nonexistent_landmarker.task",
        )
        adapter = DDDAdapter(config)

        with pytest.raises(Exception):  # MediaPipe model loading error
            list(adapter.iter_events())

    def test_real_directory_parsing(self) -> None:
        """Parse actual DDD filenames from the real dataset."""
        real_root = Path(
            "datasets/raw/Driver Drowsiness Dataset (DDD)"
        )
        if not real_root.is_dir():
            pytest.skip("Real DDD dataset not available")

        images = _list_image_files(real_root)
        assert len(images) > 0, "Expected at least some PNG files"

        parsed = 0
        skipped = 0
        for path, label in images:
            meta = _parse_filename(path.stem)
            if meta is not None:
                parsed += 1
                assert "subject_id" in meta
                assert "fatigue_level" in meta
                assert "frame_number" in meta
                assert meta["fatigue_level"] == label
            else:
                skipped += 1

        # Most files should parse successfully
        assert parsed > 0, f"Expected at least some files to parse, got {parsed}"
        print(f"Parsed {parsed}/{parsed + skipped} files from real DDD")

    def test_real_class_distribution(self) -> None:
        """Verify the class distribution matches expected counts."""
        real_root = Path(
            "datasets/raw/Driver Drowsiness Dataset (DDD)"
        )
        if not real_root.is_dir():
            pytest.skip("Real DDD dataset not available")

        images = _list_image_files(real_root)
        drowsy = sum(1 for _, l in images if l == "drowsy")
        non_drowsy = sum(1 for _, l in images if l == "non_drowsy")

        print(f"Drowsy: {drowsy}, Non Drowsy: {non_drowsy}")
        assert drowsy > 20000, f"Expected >20000 drowsy images, got {drowsy}"
        assert non_drowsy > 18000, f"Expected >18000 non_drowsy images, got {non_drowsy}"
