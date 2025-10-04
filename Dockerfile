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
    echo "import pkgutil, sys, inspect" > inspect_deepgram.py && \
    echo "found_deepgram = False" >> inspect_deepgram.py && \
    echo "for importer, modname, ispkg in pkgutil.iter_modules(sys.path):" >> inspect_deepgram.py && \
    echo "    if modname == 'deepgram':" >> inspect_deepgram.py && \
    echo "        print(f'Found deepgram module at: {importer.find_spec(modname).origin}')" >> inspect_deepgram.py && \
    echo "        found_deepgram = True" >> inspect_deepgram.py && \
    echo "        break" >> inspect_deepgram.py && \
    echo "if not found_deepgram:" >> inspect_deepgram.py && \
    echo "    print('Deepgram module not found after installation!'); sys.exit(1)" >> inspect_deepgram.py && \
    echo "" >> inspect_deepgram.py && \
    echo "import deepgram" >> inspect_deepgram.py && \
    echo "print('\\n--- Contents of deepgram module: ---')" >> inspect_deepgram.py && \
    echo "for name in dir(deepgram):" >> inspect_deepgram.py && \
    echo "    if not name.startswith('__'):" >> inspect_deepgram.py && \
    echo "        obj = getattr(deepgram, name)" >> inspect_deepgram.py && \
    echo "        if inspect.isclass(obj) and obj.__module__.startswith('deepgram'):" >> inspect_deepgram.py && \
    echo "            print(f'{name}: CLASS in {obj.__module__}.{obj.__name__}')" >> inspect_deepgram.py && \
    echo "        elif inspect.ismodule(obj) and obj.__name__.startswith('deepgram'):" >> inspect_deepgram.py && \
    echo "            print(f'{name}: SUBMODULE {obj.__name__}')" >> inspect_deepgram.py && \
    echo "        else:" >> inspect_deepgram.py && \
    echo "            print(f'{name}: {type(obj).__name__}')" >> inspect_deepgram.py && \
    echo "" >> inspect_deepgram.py && \
    echo "print('\\n--- Attempting to import DeepgramClient and PrerecordedOptions directly ---')" >> inspect_deepgram.py && \
    echo "try:" >> inspect_deepgram.py && \
    echo "    from deepgram import DeepgramClient" >> inspect_deepgram.py && \
    echo "    print('DeepgramClient found directly in deepgram package.')" >> inspect_deepgram.py && \
    echo "except ImportError:" >> inspect_deepgram.py && \
    echo "    print('DeepgramClient NOT found directly in deepgram package.')" >> inspect_deepgram.py && \
    echo "try:" >> inspect_deepgram.py && \
    echo "    from deepgram import PrerecordedOptions" >> inspect_deepgram.py && \
    echo "    print('PrerecordedOptions found directly in deepgram package.')" >> inspect_deepgram.py && \
    echo "except ImportError:" >> inspect_deepgram.py && \
    echo "    print('PrerecordedOptions NOT found directly in deepgram package.')" >> inspect_deepgram.py && \
    echo "" >> inspect_deepgram.py && \
    echo "print('\\n--- Attempting to import from deepgram.client / deepgram.options ---')" >> inspect_deepgram.py && \
    echo "try:" >> inspect_deepgram.py && \
    echo "    from deepgram.client import DeepgramClient" >> inspect_deepgram.py && \
    echo "    print('DeepgramClient found in deepgram.client.')" >> inspect_deepgram.py && \
    echo "except ImportError:" >> inspect_deepgram.py && \
    echo "    print('DeepgramClient NOT found in deepgram.client.')" >> inspect_deepgram.py && \
    echo "try:" >> inspect_deepgram.py && \
    echo "    from deepgram.options import PrerecordedOptions" >> inspect_deepgram.py && \
    echo "    print('PrerecordedOptions found in deepgram.options.')" >> inspect_deepgram.py && \
    echo "except ImportError:" >> inspect_deepgram.py && \
    echo "    print('PrerecordedOptions NOT found in deepgram.options.')" >> inspect_deepgram.py && \
    echo "" >> inspect_deepgram.py && \
    echo "print('\\n--- Deepgram module inspection complete ---')" >> inspect_deepgram.py && \
    echo "" >> inspect_deepgram.py && \
    echo "if 'DeepgramClient found directly in deepgram package.' in sys.stdout.getvalue() or \
          'DeepgramClient found in deepgram.client.' in sys.stdout.getvalue(): \
        sys.exit(0)" >> inspect_deepgram.py && \
    echo "else: sys.exit(1)" >> inspect_deepgram.py && \
    python inspect_deepgram.py && \
    rm inspect_deepgram.py || \
    (echo "!!! ERROR: Deepgram SDK inspection failed. Check above logs for details. !!!" && exit 1)

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
