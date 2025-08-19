# Stage 1: The Build Stage
# We'll use your specified base image to build the environment and install dependencies.
FROM python:3.10-slim-buster as builder

# Set the working directory for the build stage.
WORKDIR /app

# Copy the requirements file first to leverage Docker's caching.
COPY requirements.txt .

# Create and populate a virtual environment with all dependencies.
RUN python3 -m venv /venv
RUN /venv/bin/pip install --no-cache-dir --upgrade pip
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy all application source code, models, and tests into the build stage.
COPY src/ ./src/
COPY models/ ./models/
COPY tests/ ./tests/


# Stage 2: The Final Production Stage
# This stage uses a distroless base image for a tiny, secure final product.
# It contains only Python and essential C-libraries.
FROM gcr.io/distroless/python3

# Set the working directory for the final image.
WORKDIR /app

# Copy the virtual environment and the application code from the builder stage.
# We are only copying exactly what's needed for the application to run.
COPY --from=builder /venv /venv
COPY --from=builder /app .

# Set the PATH to ensure the virtual environment's binaries are used at runtime.
ENV PATH="/venv/bin:$PATH"

# Expose the specific port for the API.
EXPOSE 5001

# The final command to run the application with gunicorn.
# The path must be explicit for distroless images.
CMD ["/venv/bin/gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "--timeout", "120", "src.app:app"]
