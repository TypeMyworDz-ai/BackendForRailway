# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies. ffmpeg is used by pydub / ffmpeg-python
# to compress audio before it is sent to a transcription service.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# A current pip resolves this dependency set far more reliably than the
# one bundled with the base image.
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies before copying the app, so that changing main.py
# does not invalidate the cached dependency layer on every deploy.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application itself
COPY . /app

# Make port 8000 available (informational, Railway overrides this)
EXPOSE 8000

# Run Uvicorn directly using exec form, relying on Railway's PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
