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
# Create a temporary Python script to inspect the deepgram module
RUN echo "--- Attempting to install deepgram-sdk in isolation ---" && \
    pip install --no-cache-dir deepgram-sdk==2.12.0 && \
    echo "--- deepgram-sdk installation command finished. Creating inspection script... ---" && \
    cat > inspect_deepgram.py <<EOF
import pkgutil, sys, inspect

print('--- Deepgram Module Inspection Results ---')

# Check top-level deepgram package
try:
    import deepgram
    print(f'Found deepgram module at: {deepgram.__file__}')

    print('\nAttempting to import DeepgramClient and PrerecordedOptions:')
    
    # Try direct import from deepgram
    try:
        from deepgram import DeepgramClient
        print('    DeepgramClient found directly in deepgram package.')
        deepgram_client_found = True
    except ImportError:
        print('    DeepgramClient NOT found directly in deepgram package.')
        deepgram_client_found = False

    try:
        from deepgram import PrerecordedOptions
        print('    PrerecordedOptions found directly in deepgram package.')
        prerecorded_options_found = True
    except ImportError:
        print('    PrerecordedOptions NOT found directly in deepgram package.')
        prerecorded_options_found = False

    # Try import from deepgram.client / deepgram.options
    print('\nAttempting to import from deepgram.client / deepgram.options:')
    try:
        from deepgram.client import DeepgramClient
        print('    DeepgramClient found in deepgram.client.')
        deepgram_client_found = True
    except ImportError:
        print('    DeepgramClient NOT found in deepgram.client.')
    
    try:
        from deepgram.options import PrerecordedOptions
        print('    PrerecordedOptions found in deepgram.options.')
        prerecorded_options_found = True
    except ImportError:
        print('    PrerecordedOptions NOT found in deepgram.options.')

    # Final check and exit status
    if deepgram_client_found and prerecorded_options_found:
        print('\n--- SUCCESS: Both DeepgramClient and PrerecordedOptions found. ---')
        sys.exit(0)
    else:
        print('\n--- FAILURE: One or both Deepgram components not found. ---')
        sys.exit(1)

except ImportError as e:
    print(f'Deepgram module itself could not be imported: {e}')
    sys.exit(1)
EOF
# The line above (EOF) MUST NOT have a backslash for cat <<EOF to work.
# The backslash should be on the line *before* the cat command if you were chaining.
# However, to chain commands *after* cat <<EOF, we simply add '&& \' after the EOF.
 && \
    python inspect_deepgram.py && \
    rm inspect_deepgram.py || \
    (echo "!!! ERROR: Deepgram SDK inspection script execution failed. Check above logs for details. !!!" && exit 1)

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
