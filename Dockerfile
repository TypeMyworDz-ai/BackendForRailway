# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies. ffmpeg is used by pydub / ffmpeg-python to
# compress audio before it is sent to a transcription service, and pydub also
# shells out to ffprobe to read a file's format and duration, so BOTH binaries
# have to be present. The Debian ffmpeg package provides both.
#
# This step used to be a single plain apt-get install, and on 7 September 2026
# it brought the whole deploy down. Debian's mirror sits behind a CDN, and one
# node served an index that no longer matched the pool, so the build died with:
#
#   E: Failed to fetch .../libfftw3-double3_3.3.8-2_amd64.deb
#      Error reading from server - read (104: Connection reset by peer)
#   E: Failed to fetch .../libglx-mesa0_20.3.5-1+deb11u1_amd64.deb  404 Not Found
#   E: Unable to fetch some archives
#
# Nothing was wrong with our code; a merged pricing change simply could not
# reach production for over an hour. Installing ffmpeg pulls in 188 packages
# and 114 MB, so there are 188 chances for one flaky fetch to fail the build.
#
# Two defences, because they cover different failures:
#
#   Acquire::Retries makes apt retry an individual fetch that resets or times
#   out, which handles the connection-reset case.
#
#   The outer loop re-runs apt-get update as well, which is what handles the
#   404: a stale index is only fixed by fetching a fresh one, usually from a
#   healthier CDN node. Retrying the install alone would fail identically.
#
# Finally, the versions are asserted. If ffmpeg or ffprobe is somehow absent,
# the build stops here rather than producing an image that looks fine and then
# fails on the first client upload.
#
# The version is captured into a variable rather than piped into head. A shell
# pipeline reports the exit status of its LAST command, so "ffmpeg -version |
# head -1" succeeds even when ffmpeg does not exist -- head is perfectly happy
# reading nothing. That is not a theoretical concern: the first version of this
# check was written that way, and a test with the binary removed passed when it
# should have failed. Command substitution in an assignment does propagate the
# failure under set -e, so this form genuinely stops the build.
RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
        echo "ffmpeg install, attempt $attempt of 5"; \
        if apt-get update -o Acquire::Retries=5 \
           && apt-get install -y --no-install-recommends \
                -o Acquire::Retries=5 ffmpeg; then \
            echo "ffmpeg installed on attempt $attempt"; \
            break; \
        fi; \
        if [ "$attempt" = "5" ]; then \
            echo "ffmpeg could not be installed after 5 attempts"; \
            exit 1; \
        fi; \
        echo "attempt $attempt failed, clearing lists and retrying"; \
        rm -rf /var/lib/apt/lists/*; \
        sleep 5; \
    done; \
    FFMPEG_V="$(ffmpeg -version)"; echo "$FFMPEG_V" | head -1; \
    FFPROBE_V="$(ffprobe -version)"; echo "$FFPROBE_V" | head -1; \
    rm -rf /var/lib/apt/lists/*

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
