FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_HTTP_TIMEOUT=120
ENV UV_HTTP_RETRIES=5

COPY pyproject.toml uv.lock ./
COPY apps/api/pyproject.toml ./apps/api/
COPY apps/api/README.md ./apps/api/

RUN uv sync --frozen --no-dev --package api

COPY --chown=app:app apps/api ./apps/api

ENV PYTHONOPTIMIZE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/apps/api/src:$PYTHONPATH"

RUN addgroup --system app && \
    adduser --system --ingroup app app

USER app

EXPOSE 8000

WORKDIR /app/apps/api/src

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]