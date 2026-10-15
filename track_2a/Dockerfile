FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create non-root user for runtime
RUN addgroup --system app && adduser --system --ingroup app app

# -----------------------------
# Builder stage
# -----------------------------
FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /bin/uv

RUN apt-get -o Acquire::Retries=5 update && apt-get -o Acquire::Retries=5 install -y --no-install-recommends --fix-missing \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached layer)
COPY pyproject.toml README.md ./
COPY uv.loc[k] ./
RUN uv sync --no-dev --no-install-project --compile-bytecode

# Copy source and install project
COPY src/ ./src/
RUN uv sync --no-dev --compile-bytecode

# -----------------------------
# Runtime stage
# -----------------------------
FROM base AS final

RUN apt-get -o Acquire::Retries=5 update && apt-get -o Acquire::Retries=5 install -y --no-install-recommends --fix-missing \
    ca-certificates \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY aegra.json .
COPY src/ ./src/
COPY data/ ./data/

# make sure the non-root user can read code + data regardless of host file permissions
RUN chmod -R a+rX /app/src /app/data /app/aegra.json

ENV PATH="/app/.venv/bin:$PATH" \
    SMARTSTART_DATA_DIR=/app/data/fhgr

EXPOSE 2026

USER app

CMD ["aegra", "serve"]
