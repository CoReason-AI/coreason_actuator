# Stage 1: Builder
FROM python:3.14-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy the project files
COPY pyproject.toml .
COPY uv.lock .
COPY src/ ./src/
COPY README.md .
COPY LICENSE .

# Install dependencies and build the wheel
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --frozen

# Ensure project build step is included
RUN uv build --wheel --out-dir /wheels


# Stage 2: Runtime
FROM python:3.14-slim AS runtime

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Add user's local bin to PATH
ENV PATH="/home/appuser/app/.venv/bin:/home/appuser/.local/bin:${PATH}"

# Set the working directory
WORKDIR /home/appuser/app
COPY --from=builder --chown=appuser:appuser /app/.venv ./.venv

# Copy the wheel from the builder stage
COPY --from=builder /wheels /wheels

# Install the application wheel
RUN uv pip install --no-cache /wheels/*.whl

CMD ["python", "-m", "coreason_actuator.main"]
