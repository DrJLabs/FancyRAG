FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFER_PREBUILT=1 \
    PATH=/root/.local/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY src ./src
COPY servers ./servers
COPY queries ./queries
COPY scripts/config ./scripts/config
COPY tools ./tools
COPY README.md ./

RUN uv sync --frozen \
    && rm -rf /root/.cache/uv

ENV PATH=/app/.venv/bin:$PATH

EXPOSE 8080

ENTRYPOINT ["python", "servers/mcp_hybrid_google.py"]
