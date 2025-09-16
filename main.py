import logging
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
import tempfile
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import requests
from pydub import AudioSegment

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

# Load environment variables
logger.info("Loading environment variables...")
load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
logger.info(f"Attempted to load ASSEMBLYAI_API_KEY. Value found: {bool(ASSEMBLYAI_API_KEY)}")
if not ASSEMBLYAI_API_KEY:
    logger.error("ASSEMBLYAI_API_KEY environment variable not set!")
    sys.exit(1)
logger.info("Environment variables loaded successfully")

# Audio conversion function
def convert_to_mp3(input_path: str, output_path: str = None) -> str:
    """Convert audio file to MP3 format"""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}.mp3"
    
    try:
        logger.info(f"Converting {input_path} to MP3 format...")
        # Load audio file and convert to MP3
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="mp3", bitrate="128k")
        logger.info(f"Successfully converted to MP3: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error converting to MP3: {e}")
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
        # === AssemblyAI Integration ===
        logger.info(f"Background task: Sending audio {filename} to AssemblyAI for transcription...")
        headers = {
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/json"
        }
        
        # Upload audio file to AssemblyAI
        upload_endpoint = "https://api.assemblyai.com/v2/upload"
        with open(tmp_path, "rb") as f:
            upload_response = requests.post(upload_endpoint, headers=headers, data=f)
        
        if upload_response.status_code != 200:
            logger.error(f"Background task: AssemblyAI upload failed for {filename}: {upload_response.status_code} - {upload_response.text}")
            job_data.update({
                "status": "failed",
                "error": "Failed to upload audio to AssemblyAI",
                "completed_at": datetime.now().isoformat()
            })
            return
        
        upload_url = upload_response.json()["upload_url"]
        logger.info(f"Background task: Audio uploaded to AssemblyAI: {upload_url}")

        # Start transcription
        transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
        json_data = {
            "audio_url": upload_url,
            "language_code": "en_us"
        }
        
        transcript_response = requests.post(transcript_endpoint, headers=headers, json=json_data)
        
        if transcript_response.status_code != 200:
            logger.error(f"Background task: AssemblyAI transcription start failed for {filename}: {transcript_response.status_code} - {transcript_response.text}")
            job_data.update({
                "status": "failed",
                "error": "Failed to start transcription on AssemblyAI",
                "completed_at": datetime.now().isoformat()
            })
            return
        
        transcript_id = transcript_response.json()["id"]
        job_data["assemblyai_id"] = transcript_id
        logger.info(f"Background task: AssemblyAI transcription started with ID: {transcript_id}")
        # === End AssemblyAI Integration ===

    except Exception as e:
        logger.error(f"Background task: ERROR during AssemblyAI integration for job {job_id}: {str(e)}")
        import traceback
        logger.error(f"Background task: Full traceback: {traceback.format_exc()}")
        job_data.update({
            "status": "failed",
            "error": "Internal server error during transcription initiation",
            "completed_at": datetime.now().isoformat()
        })
    finally:
        # Clean up the temporary file in background task
        if os.path.exists(tmp_path):
            logger.info(f"Background task: Cleaning up temporary file: {tmp_path}")
            os.unlink(tmp_path)
        logger.info(f"Background task completed for job ID: {job_id}")


@app.get("/")
async def root(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    logger.info("Root endpoint called")
    return {"message": "Transcription Service is running!"}

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
        "assemblyai_id": None 
    }
    logger.info(f"Job {job_id} initialized with status 'processing'")
    
    # Save the uploaded file temporarily and convert to MP3
    try:
        logger.info(f"Saving uploaded file {file.filename} temporarily...")
        # Save original file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            original_tmp_path = tmp.name
        
        logger.info(f"Original file saved to: {original_tmp_path}")
        
        # Convert to MP3
        mp3_tmp_path = original_tmp_path.replace(os.path.splitext(original_tmp_path)[1], '.mp3')
        convert_to_mp3(original_tmp_path, mp3_tmp_path)
        
        # Clean up original file
        os.unlink(original_tmp_path)
        logger.info(f"Original file cleaned up, MP3 file ready: {mp3_tmp_path}")
        
    except Exception as e:
        logger.error(f"ERROR processing file for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process audio file")

    # Add the processing to background task with MP3 file
    background_tasks.add_task(process_transcription_job, job_id, mp3_tmp_path, file.filename)
    
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
    
    # If transcription is still processing, poll AssemblyAI for status
    if job_data["status"] == "processing" and job_data["assemblyai_id"]:
        logger.info(f"Polling AssemblyAI for status of transcript ID: {job_data['assemblyai_id']}")
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        transcript_endpoint = f"https://api.assemblyai.com/v2/transcript/{job_data['assemblyai_id']}"
        
        response_data = requests.get(transcript_endpoint, headers=headers)
        
        if response_data.status_code != 200:
            logger.error(f"AssemblyAI status check failed: {response_data.status_code} - {response_data.text}")
            job_data.update({
                "status": "failed",
                "error": "Failed to get status from AssemblyAI",
                "completed_at": datetime.now().isoformat()
            })
            return job_data
        
        assemblyai_result = response_data.json()
        
        if assemblyai_result["status"] == "completed":
            logger.info(f"AssemblyAI transcription {job_data['assemblyai_id']} completed.")
            job_data.update({
                "status": "completed",
                "transcription": assemblyai_result["text"],
                "language": assemblyai_result["language_code"],
                "completed_at": datetime.now().isoformat()
            })
        elif assemblyai_result["status"] == "failed":
            logger.error(f"AssemblyAI transcription {job_data['assemblyai_id']} failed: {assemblyai_result.get('error', 'Unknown error')}")
            job_data.update({
                "status": "failed",
                "error": assemblyai_result.get("error", "Transcription failed on AssemblyAI"),
                "completed_at": datetime.now().isoformat()
            })
        else:
            logger.info(f"AssemblyAI transcription {job_data['assemblyai_id']} status: {assemblyai_result['status']}")

    return job_data

logger.info("=== FASTAPI APPLICATION SETUP COMPLETE ===")

# Run this if the file is executed directly
if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly...")
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)