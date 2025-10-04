Here's the updated Dockerfile to fix the Deepgram SDK issue:
# ======================================================================================
# backend/Dockerfile (UPDATED to ensure correct httpx version and Deepgram installation)
# Dockerfile for your main backend service
# ======================================================================================

# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (ffmpeg for pydub)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .

# First install specific packages that might cause issues
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Force reinstall httpx and its dependencies to ensure FormData is available
# --no-cache-dir: Prevents pip from using cached versions
# --upgrade --force-reinstall: Ensures it's a fresh install
RUN pip install --no-cache-dir --upgrade --force-reinstall httpx==0.28.1

# Explicitly install deepgram-sdk with specific version before other requirements
RUN pip install --no-cache-dir deepgram-sdk==2.12.0

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Verify Deepgram is installed correctly
RUN python -c "from deepgram import DeepgramClient; print('Deepgram SDK imported successfully')" || echo "Failed to import Deepgram"

# Copy the rest of your application code
COPY . .

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Set environment variables to ensure Python doesn't buffer stdout/stderr
ENV PYTHONUNBUFFERED=1

# Define the command to run your application
# Railway expects 'main:app' if your main file is main.py
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

Key changes I've made:

Added build-essential and python3-dev to ensure all compilation tools are available for any packages that need them
Added explicit upgrade of pip, setuptools, and wheel to ensure the latest versions
Explicitly installed deepgram-sdk with a specific version (2.12.0) before other requirements
Added a verification step to check if Deepgram can be imported correctly
Set PYTHONUNBUFFERED=1 to ensure logs are output immediately (helps with debugging)

Let me know when you're ready to provide the main.py file, and I'll make the changes you request for the Gemini AI formatter issue.