# ===================================================================
# Stage 1: The "builder" stage
# We use the official uv image to install our dependencies.
# ===================================================================
FROM ghcr.io/astral-sh/uv:latest AS uv-builder

# Set the working directory inside the builder container
WORKDIR /app

# Copy the project dependency files
# This allows Docker to cache the dependency installation step
# if these files haven't changed.
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment.
# The --frozen flag tells uv to use the exact versions in uv.lock.
# This creates a `.venv` directory in /app.
#RUN uv sync --frozen


# ===================================================================
# Stage 2: The "final" stage
# This is the image that will be run in production.
# ===================================================================
# Start from the base image provided in the homework
FROM agrigorev/zoomcamp-model:2025

# Set the working directory in the final image
WORKDIR /code

# Copy the virtual environment from the "uv-builder" stage
COPY --from=uv-builder /app/.venv /opt/venv

# Copy your FastAPI application script into the image
COPY app.py .

# Update the PATH environment variable so the OS can find
# python and installed packages inside our virtual environment.
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# The command to run your application when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]