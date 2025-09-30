# ====================================================================================================
# backend/Dockerfile (UPDATED to ensure correct httpx version)
# Dockerfile for your main backend service
# ====================================================================================================

# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (ffmpeg for pydub)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .

# Force reinstall httpx and its dependencies to ensure FormData is available
# --no-cache-dir: Prevents pip from using cached versions
# --upgrade --force-reinstall: Ensures it's a fresh install
RUN pip install --no-cache-dir --upgrade --force-reinstall httpx==0.28.1 \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Define the command to run your application
# Railway expects 'main:app' if your main file is main.py
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ====================================================================================================
