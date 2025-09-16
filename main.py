import logging
import sys
import asyncio
import subprocess
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
import tempfile
import uuid
from datetime import datetime
from dotenv import load_dotenv
import requests
from pydub import AudioSegment
import assemblyai as aai

# Configure logging to be very verbose
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== STARTING FASTAPI APPLICATION ===")

# Install ffmpeg if not available
def install_ffmpeg():
    try:
        # Test if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        logger.info("ffmpeg is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("Installing ffmpeg...")
        try:
            # Try to install ffmpeg on Ubuntu/Debian (Railway uses Ubuntu)
            subprocess.run(['apt-get', 'update'], check=True)
            subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
            logger.info("ffmpeg installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install ffmpeg: {e}")
            # Continue without ffmpeg - basic conversion will still work

# Install ffmpeg on startup
install_ffmpeg()

# Load environment variables
logger.info("Loading environment variables...")
load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
logger.info(f"Attempted to load ASSEMBLYAI_API_KEY. Value found: {bool(ASSEMBLYAI_API_KEY)}")
if not ASSEMBLYAI_API_KEY:
    logger.error("ASSEMBLYAI_API_KEY environment variable not set!")
    sys.exit(1)

# Configure AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY
logger.info("Environment variables loaded successfully")
# Enhanced audio compression function
def compress_audio_for_transcription(input_path: str, output_path: str = None) -> tuple[str, dict]:
    """Compress audio file optimally for AssemblyAI transcription"""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_compressed.mp3"
    
    try:
        logger.info(f"Compressing {input_path} for transcription...")
        
        # Get original file size
        input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        logger.info(f"Original file size: {input_size:.2f} MB")
        
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        logger.info(f"Original audio: {audio.channels} channels, {audio.frame_rate}Hz, {len(audio)}ms")
        
        # Optimize for transcription (balance size vs quality):
        # 1. Convert to mono (reduces size by ~50%)
        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info("Converted to mono audio")
        
        # 2. Optimize sample rate for speech recognition
        target_sample_rate = 16000  # Optimal for speech recognition
        if audio.frame_rate != target_sample_rate:
            audio = audio.set_frame_rate(target_sample_rate)
            logger.info(f"Optimized sample rate to {target_sample_rate} Hz")
        
        # 3. Apply audio normalization for better transcription
        normalized_audio = audio.normalize()
        logger.info("Applied audio normalization")
        
        # 4. Export with optimized settings for transcription
        try:
            normalized_audio.export(
                output_path, 
                format="mp3",
                bitrate="64k",  # Good balance for speech
                parameters=[
                    "-q:a", "2",    # Good quality for speech
                    "-ac", "1",     # Force mono
                    "-ar", str(target_sample_rate),  # Force sample rate
                    "-af", "highpass=f=80,lowpass=f=8000"  # Filter for speech frequencies
                ]
            )
            logger.info("Used optimized compression with audio filtering")
        except Exception as ffmpeg_error:
            logger.warning(f"Advanced compression failed: {ffmpeg_error}")
            # Fallback to basic export
            normalized_audio.export(output_path, format="mp3", bitrate="64k")
            logger.info("Used basic compression fallback")
        
        # Calculate compression stats
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            compression_ratio = (1 - output_size/input_size) * 100 if input_size > 0 else 0
            
            stats = {
                "original_size_mb": round(input_size, 2),
                "compressed_size_mb": round(output_size, 2),
                "compression_ratio_percent": round(compression_ratio, 1),
                "duration_seconds": len(audio) / 1000.0
            }
            
            logger.info(f"Compression complete:")
            logger.info(f"  Original: {stats['original_size_mb']} MB")
            logger.info(f"  Compressed: {stats['compressed_size_mb']} MB")
            logger.info(f"  Size reduction: {stats['compression_ratio_percent']}%")
            logger.info(f"  Duration: {stats['duration_seconds']:.1f} seconds")
        
        return output_path, stats
        
    except Exception as e:
        logger.error(f"Error compressing audio: {e}")
        raise

def compress_audio_for_download(input_path: str, output_path: str = None, quality: str = "high") -> str:
    """Compress audio file for download with different quality options"""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_download.mp3"
    
    try:
        logger.info(f"Compressing {input_path} for download (quality: {quality})...")
        
        audio = AudioSegment.from_file(input_path)
        
        # Quality settings
        if quality == "high":
            bitrate = "192k"
            sample_rate = 44100
            channels = 2 if audio.channels > 1 else 1
        elif quality == "medium":
            bitrate = "128k"
            sample_rate = 44100
            channels = 2 if audio.channels > 1 else 1
        else:  # low quality
            bitrate = "96k"
            sample_rate = 22050
            channels = 1
        
        # Apply settings
        if audio.channels != channels:
            audio = audio.set_channels(channels)
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)
        
        # Export with settings
        audio.export(
            output_path,
            format="mp3",
            bitrate=bitrate,
            parameters=[
                "-q:a", "2",
                "-ac", str(channels),
                "-ar", str(sample_rate)
            ]
        )
        
        logger.info(f"Download compression complete: {quality} quality")
        return output_path
        
    except Exception as e:
        logger.error(f"Error compressing audio for download: {e}")
        raise
# Background task to monitor application health
async def health_monitor():
    logger.info("Starting health monitor background task")
    while True:
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            logger.info(f"Health Check - Memory: {memory_info.percent}% used, CPU: {cpu_percent}%, Available RAM: {memory_info.available / (1024**3):.2f} GB")
            await asyncio.sleep(30)  # Log every 30 seconds
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
            await asyncio.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application lifespan startup")
    # Start background health monitoring
    health_task = asyncio.create_task(health_monitor())
    logger.info("Health monitor task created")
    yield
    # Shutdown
    logger.info("Application lifespan shutdown")
    health_task.cancel()

# Create the FastAPI app
logger.info("Creating FastAPI app...")
app = FastAPI(title="Transcription Service", lifespan=lifespan)
logger.info("FastAPI app created successfully")

# Add CORS middleware
logger.info("Setting up CORS middleware with explicit origins...")
origins = [
    "http://localhost:3000",
    "https://typemywordzspeechai.vercel.app",
    "https://typemywordzspeechai-o03e6tjj3-james-gitukus-projects.vercel.app",
    "https://typemywordzspeechai-*.vercel.app",
    "https://*.vercel.app",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured successfully")

# Store transcription jobs in memory
jobs = {}
logger.info("Jobs dictionary initialized")
# Background task to handle AssemblyAI transcription
async def process_transcription_job(job_id: str, tmp_path: str, filename: str):
    logger.info(f"Background task started for job ID: {job_id}")
    job_data = jobs[job_id]

    try:
        # === Enhanced AssemblyAI Integration with Compression ===
        logger.info(f"Background task: Processing audio {filename} for transcription...")
        
        # Compress audio for optimal transcription
        compressed_path, compression_stats = compress_audio_for_transcription(tmp_path)
        
        # Update job with compression stats
        job_data["compression_stats"] = compression_stats
        
        # Upload compressed audio to AssemblyAI using the new SDK
        logger.info("Uploading compressed audio to AssemblyAI...")
        audio_url = aai.upload_file(compressed_path)
        logger.info(f"Audio uploaded to AssemblyAI: {audio_url}")

        # Configure transcription settings
        config = aai.TranscriptionConfig(
            audio_url=audio_url,
            language_code="en_us",
            punctuate=True,
            format_text=True,
            speaker_labels=False,  # Can be enabled if needed
            auto_highlights=False,
            sentiment_analysis=False
        )
        
        # Start transcription
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(config)
        
        if transcript.status == aai.TranscriptStatus.error:
            logger.error(f"AssemblyAI transcription failed: {transcript.error}")
            job_data.update({
                "status": "failed",
                "error": f"Transcription failed: {transcript.error}",
                "completed_at": datetime.now().isoformat()
            })
        else:
            logger.info(f"Transcription completed successfully for job {job_id}")
            job_data.update({
                "status": "completed",
                "transcription": transcript.text,
                "language": "en_us",
                "confidence": transcript.confidence if hasattr(transcript, 'confidence') else None,
                "completed_at": datetime.now().isoformat()
            })
        
        # Clean up compressed file
        if os.path.exists(compressed_path):
            os.unlink(compressed_path)
            logger.info(f"Cleaned up compressed file: {compressed_path}")
        
        # === End Enhanced AssemblyAI Integration ===

    except Exception as e:
        logger.error(f"Background task: ERROR during transcription for job {job_id}: {str(e)}")
        import traceback
        logger.error(f"Background task: Full traceback: {traceback.format_exc()}")
        job_data.update({
            "status": "failed",
            "error": "Internal server error during transcription",
            "completed_at": datetime.now().isoformat()
        })
    finally:
        # Clean up the original temporary file
        if os.path.exists(tmp_path):
            logger.info(f"Background task: Cleaning up original temporary file: {tmp_path}")
            os.unlink(tmp_path)
        logger.info(f"Background task completed for job ID: {job_id}")
@app.get("/")
async def root(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    logger.info("Root endpoint called")
    return {"message": "Enhanced Transcription Service with Audio Compression is running!"}

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks(), response: Response = Response()):
    response.headers["Access-Control-Allow-Origin"] = "*"
    logger.info(f"Transcribe endpoint called with file: {file.filename}")
    
    # Check if file is audio or video
    if not file.content_type.startswith(('audio/', 'video/')):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Please upload an audio or video file")
    
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"Created job ID: {job_id}")
    
    # Initialize job status
    jobs[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "created_at": datetime.now().isoformat(),
        "assemblyai_id": None,
        "compression_stats": None
    }
    logger.info(f"Job {job_id} initialized with status 'processing'")
    
    # Save the uploaded file temporarily
    try:
        logger.info(f"Saving uploaded file {file.filename} temporarily...")
        # Save original file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"File saved to: {tmp_path}")
        
    except Exception as e:
        logger.error(f"ERROR processing file for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process audio file")

    # Add the processing to background task
    background_tasks.add_task(process_transcription_job, job_id, tmp_path, file.filename)
    
    # Return immediate response
    logger.info(f"Returning immediate response for job ID: {job_id}")
    return {"job_id": job_id, "status": jobs[job_id]["status"]}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str, response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    logger.info(f"Status check for job ID: {job_id}")
    if job_id not in jobs:
        logger.warning(f"Job ID {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs[job_id]
    logger.info(f"Returning status for job {job_id}: {job_data['status']}")
    return job_data

@app.post("/compress-download")
async def compress_download(file: UploadFile = File(...), quality: str = "high", response: Response = Response()):
    """Endpoint to compress audio files for download"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    logger.info(f"Compress download endpoint called with file: {file.filename}, quality: {quality}")
    
    if not file.content_type.startswith(('audio/', 'video/')):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Please upload an audio or video file")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            input_path = tmp.name
        
        # Compress for download
        output_path = compress_audio_for_download(input_path, quality=quality)
        
        # Read compressed file
        with open(output_path, 'rb') as f:
            compressed_content = f.read()
        
        # Clean up files
        os.unlink(input_path)
        os.unlink(output_path)
        
        # Return compressed file
        from fastapi.responses import Response as FastAPIResponse
        return FastAPIResponse(
            content=compressed_content,
            media_type="audio/mp3",
            headers={"Content-Disposition": f"attachment; filename=compressed_{file.filename}.mp3"}
        )
        
    except Exception as e:
        logger.error(f"Error compressing file for download: {e}")
        raise HTTPException(status_code=500, detail="Failed to compress audio file")

logger.info("=== FASTAPI APPLICATION SETUP COMPLETE ===")

# Run this if the file is executed directly
if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly...")
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)