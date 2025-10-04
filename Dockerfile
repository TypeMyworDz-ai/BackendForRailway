# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (ffmpeg for pydub and build tools)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel first to ensure a robust installation environment
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# --- Install ALL Python dependencies from requirements.txt (No filtering, no individual installs) ---
# This is the single source of truth for Python packages.
RUN echo "--- Installing ALL Python dependencies from requirements.txt ---" && \
    pip install --no-cache-dir -r requirements.txt

# --- DIAGNOSTIC STEP 1: Verify Deepgram base import ---
RUN echo "--- Verifying Deepgram SDK (base module) ---" && \
    python -c "import deepgram; print('Deepgram SDK (base module) imported successfully.')" || \
    (echo "!!! ERROR: Deepgram SDK (base module) failed to import. Check above logs for details. !!!" && exit 1)

# --- DIAGNOSTIC STEP 2: Verify Uvicorn import ---
RUN echo "--- Verifying Uvicorn module ---" && \
    python -c "import uvicorn; print('Uvicorn module imported successfully.')" || \
    (echo "!!! ERROR: Uvicorn module failed to import. Check above logs for details. !!!" && exit 1)

# --- DIAGNOSTIC STEP 3: Verify FastAPI import ---
RUN echo "--- Verifying FastAPI module ---" && \
    python -c "import fastapi; print('FastAPI module imported successfully.')" || \
    (echo "!!! ERROR: FastAPI module failed to import. Check above logs for details. !!!" && exit 1)

# --- DIAGNOSTIC STEP 4: Verify Requests import (NEW) ---
RUN echo "--- Verifying Requests module ---" && \
    python -c "import requests; print('Requests module imported successfully.')" || \
    (echo "!!! ERROR: Requests module failed to import. Check above logs for details. !!!" && exit 1)

# Check for any broken dependencies (this can sometimes reveal conflicts)
RUN echo "--- Running pip check for broken dependencies ---" && \
    pip check || \
    (echo "!!! WARNING: pip check found broken dependencies. See above for details. !!!" && true)


# Copy the rest of your application code
COPY . .

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Set environment variables to ensure Python doesn't buffer stdout/stderr
ENV PYTHONUNBUFFERED=1

# Define the command to run your application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
