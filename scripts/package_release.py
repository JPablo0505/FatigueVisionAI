#!/usr/bin/env python3
"""Automated release packaging for FatigueVision AI.

Orchestrates:
  1. Frontend build (Vite → static files)
  2. Backend build (PyInstaller → portable exe)
  3. Inno Setup compilation (optional, generates .exe installer)

Usage:
    python scripts/package_release.py              # full pipeline
    python scripts/package_release.py --skip-inno  # skip installer step
    python scripts/package_release.py --frontend-only  # just build frontend
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "apps" / "web"
BACKEND_DIR = REPO_ROOT / "apps" / "backend"
SPEC_FILE = BACKEND_DIR / "fatigue_backend.spec"
DIST_DIR = REPO_ROOT / "dist"
RELEASE_DIR = DIST_DIR / "release"
INNO_SCRIPT = REPO_ROOT / "installer.iss"
VERSION = "0.1.0"


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and stream output."""
    print(f"\n{'='*60}")
    print(f"  > {' '.join(cmd)}")
    print(f"  in {cwd or Path.cwd()}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if check and result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def build_frontend() -> None:
    """Build the Vite/React frontend into static files."""
    print("\n[1/4] Building frontend (Vite)...")
    _run(["npm", "run", "build"], cwd=FRONTEND_DIR)

    dist = FRONTEND_DIR / "dist"
    if not dist.exists():
        print("[ERROR] Frontend dist/ not found after build")
        sys.exit(1)
    print(f"  ✓ Frontend built → {dist}")


def build_backend() -> None:
    """Build the backend executable with PyInstaller."""
    print("\n[2/4] Building backend (PyInstaller)...")

    if not SPEC_FILE.exists():
        print(f"[ERROR] Spec file not found: {SPEC_FILE}")
        sys.exit(1)

    _run([sys.executable, "-m", "PyInstaller", str(SPEC_FILE)], cwd=REPO_ROOT)

    exe_dir = DIST_DIR / "fatigue_backend"
    exe_file = exe_dir / "fatigue_backend.exe"
    if not exe_file.exists():
        print(f"[ERROR] Backend exe not found: {exe_file}")
        sys.exit(1)
    print(f"  ✓ Backend built → {exe_dir}")


def assemble_release() -> None:
    """Copy the built artifacts into a clean release directory."""
    print("\n[3/4] Assembling release directory...")

    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True)

    # Copy backend dist
    backend_src = DIST_DIR / "fatigue_backend"
    backend_dst = RELEASE_DIR / "FatigueVisionAI"
    shutil.copytree(backend_src, backend_dst)
    print(f"  ✓ Backend → {backend_dst}")

    # Create a launcher script
    launcher = RELEASE_DIR / "FatigueVisionAI.bat"
    launcher.write_text(
        '@echo off\n'
        'echo Starting FatigueVision AI...\n'
        'echo.\n'
        'echo The dashboard will open at http://localhost:8000\n'
        'echo Press Ctrl+C to stop.\n'
        'echo.\n'
        'set CAMERA_SOURCE=browser\n'
        'set FATIGUE_MODEL_TYPE=ml\n'
        'start "" http://localhost:8000\n'
        '"FatigueVisionAI\\fatigue_backend.exe"\n',
        encoding="utf-8",
    )
    print(f"  ✓ Launcher → {launcher}")

    # Create a README
    readme = RELEASE_DIR / "README.txt"
    readme.write_text(
        f"FatigueVision AI v{VERSION}\n"
        f"{'='*40}\n\n"
        "1. Double-click FatigueVisionAI.bat to start\n"
        "2. The dashboard opens automatically at http://localhost:8000\n"
        "3. Allow camera access in your browser when prompted\n"
        "4. Click Start to begin fatigue monitoring\n\n"
        "Requirements:\n"
        "- A webcam or browser with camera access\n"
        "- Modern browser (Chrome, Edge, Firefox)\n\n"
        "Troubleshooting:\n"
        "- If the browser doesn't open, go to http://localhost:8000 manually\n"
        "- If camera doesn't work, check browser permissions\n"
        "- Backend logs are displayed in the console window\n",
        encoding="utf-8",
    )
    print(f"  ✓ README → {readme}")

    # Zip the release
    zip_path = DIST_DIR / f"FatigueVisionAI-{VERSION}-win64"
    shutil.make_archive(str(zip_path), "zip", RELEASE_DIR)
    print(f"  ✓ Zip → {zip_path}.zip")


def build_installer() -> None:
    """Build the Inno Setup installer if iscc is available."""
    print("\n[4/4] Building installer (Inno Setup)...")

    if not INNO_SCRIPT.exists():
        print(f"  ⚠ Inno Setup script not found: {INNO_SCRIPT}")
        print("  Skipping installer build.")
        return

    # Try to find iscc.exe
    iscc = shutil.which("iscc")
    if iscc is None:
        # Common install location
        iscc_path = Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe")
        if iscc_path.exists():
            iscc = str(iscc_path)

    if iscc is None:
        print("  ⚠ Inno Setup (iscc) not found in PATH")
        print("  Install from: https://jrsoftware.org/isinfo.php")
        print("  Skipping installer build.")
        return

    _run([iscc, str(INNO_SCRIPT)], cwd=REPO_ROOT)
    print(f"  ✓ Installer built")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package FatigueVision AI for release")
    parser.add_argument("--skip-inno", action="store_true", help="Skip Inno Setup step")
    parser.add_argument("--frontend-only", action="store_true", help="Only build frontend")
    args = parser.parse_args()

    print(f"FatigueVision AI Release Packager v{VERSION}")
    print(f"Repository: {REPO_ROOT}")

    build_frontend()

    if args.frontend_only:
        print("\nDone (frontend only).")
        return

    build_backend()
    assemble_release()

    if not args.skip_inno:
        build_installer()

    print(f"\n{'='*60}")
    print(f"  Release packaging complete!")
    print(f"  Artifacts in: {DIST_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
