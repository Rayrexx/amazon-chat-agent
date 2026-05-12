FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# -----------------------------
# uv + python settings
# -----------------------------
ENV UV_COMPILE_BYTECODE=1
ENV UV_HTTP_TIMEOUT=120
ENV UV_HTTP_RETRIES=5

ENV PYTHONOPTIMIZE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# -----------------------------
# Copy dependency metadata FIRST
# (important for Docker cache)
# -----------------------------
COPY pyproject.toml uv.lock ./
COPY apps/chatbot_ui/pyproject.toml ./apps/chatbot_ui/
COPY apps/chatbot_ui/README.md ./apps/chatbot_ui/

# -----------------------------
# Install dependencies
# -----------------------------
RUN uv sync --frozen --no-dev --package chatbot-ui

# -----------------------------
# Copy actual source code
# -----------------------------
COPY --chown=app:app apps/chatbot_ui ./apps/chatbot_ui

# -----------------------------
# Python paths
# -----------------------------
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/apps/chatbot_ui/src:$PYTHONPATH"

# -----------------------------
# Create non-root user
# -----------------------------
RUN addgroup --system app && \
    adduser --system --ingroup app --home /home/app app && \
    mkdir -p /home/app/.streamlit/data && \
    mkdir -p /home/app/.streamlit/cache

ENV HOME=/home/app

# -----------------------------
# Switch user
# -----------------------------
USER app

# -----------------------------
# Expose Streamlit port
# -----------------------------
EXPOSE 8501

WORKDIR /app/apps/chatbot_ui/src

# -----------------------------
# Run Streamlit
# -----------------------------
CMD ["streamlit", "run", "chatbot_ui/app.py", "--server.address=0.0.0.0"]