FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv directly from the official image binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install build dependencies if needed for C-extension compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition files first for optimal layer caching
COPY pyproject.toml uv.lock ./

# Freeze dependencies into a standalone virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# STAGE 2: Runtime
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment and project code
COPY --from=builder /app/.venv /app/.venv
COPY . /app

# Non-root security setup
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Exec form ensures proper OS signal handling (SIGTERM/SIGINT)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "9022"]
