# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for Whisper (like ffmpeg)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch separately to avoid CUDA dependencies
# This URL is for Python 3.10 and CPU. Verify on pytorch.org if your Python version changes.
RUN pip install --no-cache-dir torch==2.1.1+cpu torchvision==0.16.1+cpu torchaudio==2.1.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available (informational, Railway overrides this)
EXPOSE 8000

# Run Uvicorn directly using exec form, relying on Railway's PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]