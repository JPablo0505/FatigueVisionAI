# Build context is the repository root (see docker/compose.yaml).
FROM python:3.12-slim

# Copy the uv binary from its official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Sync dependencies from the frozen lockfile, skipping dev dependencies.
COPY apps/backend/pyproject.toml apps/backend/uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code only (tests and .env are never part of the image).
COPY apps/backend/app/ ./app/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]