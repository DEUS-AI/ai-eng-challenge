# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies into a venv inside /app/.venv
RUN uv sync --no-install-project --no-dev

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.14-slim

WORKDIR /app

# Copy the venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ ./src/

# Make sure the venv is used for all python/uvicorn calls
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
