# scripts

Scripts de desarrollo y operación del repo.

## fetch_models.py

Descarga y verifica los modelos de IA declarados en `scripts/models_manifest.json`. Solo usa la librería estándar de Python y no requiere dependencias externas.

Desde la raíz del repo:

```
uv run --no-project --python 3.12 scripts/fetch_models.py
```

Comportamiento:

- Los modelos se guardan en `models/<subdir>/<filename>` según el manifest. El subdirectorio por defecto es `vision`.
- Si el archivo ya existe y su SHA-256 coincide con el manifest, se salta (`[ok]`).
- Si falta o está corrupto, descarga a `<archivo>.part` en bloques de 1 MB, verifica el hash y mueve el archivo de forma atómica. Si el hash del `.part` no coincide, lo elimina y reporta el error.
- `uv run fetch_models.py --check`: solo verifica los archivos existentes, sin descargar. Útil para CI. Sale con código 1 si hay modelos faltantes o corruptos.
- `uv run fetch_models.py --force`: re-descarga todos los modelos aunque el hash ya sea válido.

Exit codes: `0` todo correcto, `1` algún modelo faltante/corrupto o error de descarga, `2` uso incorrecto (manejado por argparse).

### Agregar un modelo nuevo

Editar `models_manifest.json` y añadir una entrada con `name`, `subdir`, `filename`, `url` y el `sha256` real del artefacto (calculable con `Get-FileHash -Algorithm SHA256 -LiteralPath <archivo>` en PowerShell).