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

# --- START Deepgram Fix for Main Backend ---
# Explicitly uninstall deepgram-sdk first to clear any corrupted/conflicting installs.
# '|| true' prevents the build from failing if the package isn't found initially.
RUN echo "--- Attempting to uninstall deepgram-sdk (if present) ---" && \
    pip uninstall -y deepgram-sdk || true 

# --- Install ALL Python dependencies from requirements.txt (No filtering, no individual installs) ---
# Force a clean re-installation of all requirements, ensuring deepgram-sdk is fresh.
# '--no-cache-dir' prevents pip from using any local cache.
# '--force-reinstall' ensures existing packages are reinstalled.
RUN echo "--- Installing ALL Python dependencies from requirements.txt with force-reinstall ---" && \
    pip install --no-cache-dir --force-reinstall -r requirements.txt
# --- END Deepgram Fix for Main Backend ---

# --- NEW DIAGNOSTIC STEP: Inspect Deepgram module ---
# This will print the location and contents of the 'deepgram' module Python finds
RUN echo "--- DIAGNOSTIC: Inspecting 'deepgram' module ---" && \
    python -c "import deepgram; print(f'Deepgram module found at: {deepgram.__file__}'); print(f'Deepgram module contents: {dir(deepgram)}') if hasattr(deepgram, '__file__') else print('Deepgram module is a built-in or namespace package.')" || \
    (echo "!!! DIAGNOSTIC ERROR: Could not import or inspect 'deepgram' module. !!!" && exit 1)


# --- DIAGNOSTIC STEP 1: Verify Deepgram base import (Original check, kept for consistency) ---
RUN echo "--- Verifying Deepgram SDK (base module) ---" && \
    python -c "from deepgram import DeepgramClient, PrerecordedOptions; print('Deepgram SDK (DeepgramClient, PrerecordedOptions) imported successfully.')" || \
    (echo "!!! ERROR: Deepgram SDK (DeepgramClient, PrerecordedOptions) failed to import. Check above logs for details. !!!" && exit 1)

# --- DIAGNOSTIC STEP 2: Verify Uvicorn import ---
RUN echo "--- Verifying Uvicorn module ---" && \
    python -c "import uvicorn; print('Uvicorn module imported successfully.')" || \
    (echo "!!! ERROR: Uvicorn module failed to import. Check above logs for details. !!!" && exit 1)

# --- DIAGNOSTIC STEP 3: Verify FastAPI import ---
RUN echo "--- Verifying FastAPI module ---" && \
    python -c "import fastapi; print('FastAPI module imported successfully.')" || \
    (echo "!!! ERROR: FastAPI module failed to import. Check above logs for details. !!!" & && exit 1)

# --- DIAGNOSTIC STEP 4: Verify Requests import ---
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
