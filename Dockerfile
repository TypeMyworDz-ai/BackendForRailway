# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (ffmpeg for pydub and build tools)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel first to ensure a robust installation environment
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# --- Install ALL Python dependencies from requirements.txt ---
# Force a clean re-installation of all requirements, ensuring no cache is used.
RUN echo "--- Installing ALL Python dependencies from requirements.txt with force-reinstall ---" && \
    pip install --no-cache-dir --force-reinstall -r requirements.txt

# --- DIAGNOSTIC STEP: Verify essential modules (kept for general sanity check) ---
RUN echo "--- Verifying essential modules ---" && \
    python -c "import uvicorn; import fastapi; import requests; print('Uvicorn, FastAPI, and Requests modules imported successfully.')" || \
    (echo "!!! ERROR: Essential modules failed to import. Check above logs for details. !!!" && exit 1)

# Check for any broken dependencies (this can sometimes reveal conflicts)
RUN echo "--- Running pip check for broken dependencies ---" && \
    pip check || \
    (echo "!!! WARNING: pip check found broken dependencies. See above for details. !!!" && true)


# Copy the rest of your application code
COPY . /app

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Set environment variables to ensure Python doesn't buffer stdout/stderr
ENV PYTHONUNBUFFERED=1

# Define the command to run your application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
