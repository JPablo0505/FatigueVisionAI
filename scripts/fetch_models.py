"""Fetch and verify ML models declared in the model manifest.

This script is intentionally stdlib-only so it can run anywhere (CI, dev
machines) with the same interpreter used for the repo. It reads
`models_manifest.json`, ensures each model exists at `<base_dir>/<subdir>/<filename>`
and matches its declared SHA-256, and re-downloads anything missing or corrupt.

Downloads are written to `<dest>.part` in 1 MB chunks and only atomically
moved into place (`os.replace`) once their hash matches the manifest, so a
partial or corrupt download never leaves a broken file behind.

Message strings are in Spanish to match the repo language; code comments are
in English.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Spanish messages contain non-ASCII characters; force UTF-8 on Windows
# consoles/pipes so accents and the em dash are preserved.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

USER_AGENT = "FatigueVisionAI/fetch_models (model provisioning)"
CHUNK_SIZE = 1024 * 1024  # 1 MB
DOWNLOAD_TIMEOUT = 30  # seconds

MANIFEST_FILENAME = "models_manifest.json"
DEFAULT_SUBDIR = "vision"

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "scripts" / MANIFEST_FILENAME
BASE_DIR = "models"  # relative to ROOT, read from the manifest at runtime


class ModelSpec:
    """Resolved target for a single model entry in the manifest."""

    def __init__(self, name: str, subdir: str, filename: str, url: str, sha256: str):
        self.name = name
        self.subdir = subdir or DEFAULT_SUBDIR
        self.filename = filename
        self.url = url
        self.sha256 = sha256.lower()

    @property
    def destination(self) -> Path:
        return ROOT / BASE_DIR / self.subdir / self.filename


def _load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print(message: str) -> None:
    print(message)


def _print_error(message: str) -> None:
    print(message, file=sys.stderr)


def verify_file(path: Path, expected_sha256: str) -> bool:
    """Return True when the file exists and matches the expected hash."""
    if not path.is_file():
        return False
    return _sha256_of(path) == expected_sha256


def download_model(spec: ModelSpec) -> bool:
    """Download a model with hash verification.

    Streams to `<dest>.part`, verifies its SHA-256, then atomically replaces
    the final file. Returns True on success; prints the error reason on
    failure.
    """
    part_path = spec.destination.with_suffix(spec.destination.suffix + ".part")
    part_path.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            with part_path.open("wb") as out_fh:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out_fh.write(chunk)
    except urllib.error.HTTPError as exc:
        part_path.unlink(missing_ok=True)
        _print_error(f"[error] {spec.name} — error HTTP {exc.code} al descargar {spec.url}")
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        part_path.unlink(missing_ok=True)
        _print_error(f"[error] {spec.name} — error de red al descargar {spec.url}: {exc}")
        return False

    part_sha = _sha256_of(part_path)
    if part_sha != spec.sha256:
        part_path.unlink(missing_ok=True)
        _print_error(
            f"[error] {spec.name} — hash no coincide (esperado {spec.sha256}, obtenido {part_sha}). "
            "Archivo temporal eliminado."
        )
        return False

    os.replace(part_path, spec.destination)
    return True


def cmd_fetch(specs: list[ModelSpec], force: bool) -> int:
    """Download/verify all models. Returns the process exit code."""
    ok = True
    for spec in specs:
        if not force and verify_file(spec.destination, spec.sha256):
            _print(f"[ok] {spec.name} — ya presente (hash válido)")
            continue
        if not force:
            _print(f"[descargando] {spec.name} — {spec.url}")
        else:
            _print(f"[descargando] {spec.name} (fuerza) — {spec.url}")
        if download_model(spec):
            _print(f"[ok] {spec.name} — descargado y verificado")
        else:
            ok = False
    return 0 if ok else 1


def cmd_check(specs: list[ModelSpec]) -> int:
    """Verify-only mode for CI: report missing/corrupt models, no download."""
    problems: list[str] = []
    for spec in specs:
        if verify_file(spec.destination, spec.sha256):
            _print(f"[ok] {spec.name} — hash válido")
        else:
            problems.append(spec.name)
            _print(f"[error] {spec.name} — faltante o hash inválido ({spec.destination})")

    if problems:
        _print_error("[check] modelos faltantes o corruptos: " + ", ".join(problems))
        return 1
    _print("[check] todos los modelos presentes y con hash válido")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_models.py",
        description=(
            "Descarga y verifica los modelos de IA declarados en "
            "scripts/models_manifest.json."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verifica los modelos existentes sin descargar (para CI). "
        "Sale con código 1 si hay modelos faltantes o corruptos.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-descarga todos los modelos aunque su hash ya sea válido.",
    )
    args = parser.parse_args(argv)

    manifest = _load_manifest()
    global BASE_DIR
    BASE_DIR = manifest["base_dir"]
    specs = [
        ModelSpec(
            name=entry["name"],
            subdir=entry.get("subdir", DEFAULT_SUBDIR),
            filename=entry["filename"],
            url=entry["url"],
            sha256=entry["sha256"],
        )
        for entry in manifest["models"]
    ]

    if args.check:
        return cmd_check(specs)

    return cmd_fetch(specs, force=args.force)


if __name__ == "__main__":
    sys.exit(main())