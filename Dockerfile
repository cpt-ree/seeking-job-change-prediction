
FROM python:3.13.5-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-install-project

COPY predict.py model.bin ./

CMD ["uv", "run", "--no-dev", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
