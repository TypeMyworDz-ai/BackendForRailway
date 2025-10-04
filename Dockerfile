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

# --- DIAGNOSTIC STEP 1: Install Deepgram SDK in isolation and verify base import ---
RUN echo "--- Attempting to install deepgram-sdk in isolation ---" && \
    pip install --no-cache-dir deepgram-sdk==2.12.0 && \
    echo "--- deepgram-sdk installation command finished. Verifying base import... ---" && \
    python -c "import deepgram; print('Deepgram SDK (base module) imported successfully.')" || \
    (echo "!!! ERROR: Deepgram SDK (base module) failed to import after isolated install. Check above logs for details. !!!" && exit 1)

# Install the rest of the Python dependencies from requirements.txt
# We'll filter out deepgram-sdk from requirements.txt to avoid re-installing
RUN echo "--- Installing remaining requirements from requirements.txt ---" && \
    grep -v deepgram-sdk requirements.txt > /tmp/filtered_requirements.txt && \
    pip install --no-cache-dir -r /tmp/filtered_requirements.txt && \
    rm /tmp/filtered_requirements.txt

# --- DIAGNOSTIC STEP 2: Verify Deepgram again after all installs (base import) ---
RUN echo "--- Verifying Deepgram SDK (base module) again after all requirements.txt installs ---" && \
    python -c "import deepgram; print('Deepgram SDK (base module) imported successfully (post-all-install).')" || \
    (echo "!!! ERROR: Deepgram SDK (base module) failed to import after all other installs. A dependency conflict might exist. !!!" && exit 1)

# --- NEW DIAGNOSTIC STEP 3: Verify Uvicorn import ---
RUN echo "--- Verifying Uvicorn import ---" && \
    python -c "import uvicorn; print('Uvicorn module imported successfully.')" || \
    (echo "!!! ERROR: Uvicorn module failed to import. Check above logs for details. !!!" && exit 1)

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
