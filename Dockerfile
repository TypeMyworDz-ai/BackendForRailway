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

# --- DIAGNOSTIC STEP 1: Install Deepgram SDK in isolation and inspect ---
RUN echo "--- Attempting to install deepgram-sdk in isolation ---" && \
    pip install --no-cache-dir deepgram-sdk==2.12.0 && \
    echo "--- deepgram-sdk installation command finished. Inspecting deepgram module... ---" && \
    python -c "import pkgutil; import sys; \
               found_deepgram = False; \
               for importer, modname, ispkg in pkgutil.iter_modules(sys.path): \
                   if modname == 'deepgram': \
                       print(f'Found deepgram module at: {importer.find_spec(modname).origin}'); \
                       found_deepgram = True; \
                       break; \
               if not found_deepgram: \
                   print('Deepgram module not found after installation!'); exit(1); \
               \
               import deepgram; \
               print('\n--- Contents of deepgram module: ---'); \
               for name in dir(deepgram): \
                   if not name.startswith('__'): \
                       obj = getattr(deepgram, name); \
                       if hasattr(obj, '__module__') and obj.__module__.startswith('deepgram'): \
                           print(f'{name}: {obj.__module__}.{obj.__name__}'); \
                       else: \
                           print(f'{name}: {type(obj).__name__}'); \
               print('\n--- Attempting to import DeepgramClient and PrerecordedOptions directly ---'); \
               try: \
                   from deepgram import DeepgramClient; \
                   print('DeepgramClient found directly in deepgram package.'); \
               except ImportError: \
                   print('DeepgramClient NOT found directly in deepgram package.'); \
               try: \
                   from deepgram import PrerecordedOptions; \
                   print('PrerecordedOptions found directly in deepgram package.'); \
               except ImportError: \
                   print('PrerecordedOptions NOT found directly in deepgram package.'); \
               \
               print('\n--- Attempting to import from deepgram.client ---'); \
               try: \
                   from deepgram.client import DeepgramClient; \
                   print('DeepgramClient found in deepgram.client.'); \
               except ImportError: \
                   print('DeepgramClient NOT found in deepgram.client.'); \
               try: \
                   from deepgram.options import PrerecordedOptions; \
                   print('PrerecordedOptions found in deepgram.options.'); \
               except ImportError: \
                   print('PrerecordedOptions NOT found in deepgram.options.'); \
               \
               print('\n--- Deepgram module inspection complete ---'); \
               " || (echo "!!! ERROR: Deepgram SDK inspection failed. Check above logs for details. !!!" && exit 1)

# Install the rest of the Python dependencies from requirements.txt
# We'll filter out deepgram-sdk from requirements.txt to avoid re-installing
RUN echo "--- Installing remaining requirements from requirements.txt ---" && \
    pip install --no-cache-dir -r <(grep -v deepgram-sdk requirements.txt)

# --- DIAGNOSTIC STEP 2: Verify Deepgram again after all installs (using new knowledge) ---
# This step will be updated once we know the correct import paths from Step 1's logs.
# For now, we'll keep a placeholder or re-run the inspection.
RUN echo "--- Final Deepgram verification (placeholder) ---" && \
    python -c "print('Deepgram SDK final verification will be updated after inspection.')"


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
