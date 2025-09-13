import logging
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Response # Import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
import tempfile
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import requests

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
logger.info(f"Attempted to load ASSEMBLYAI_API_KEY. Value found: {bool(ASSEMBLYAI_API_KEY)}") # Debug line
if not ASSEMBLYAI_API_KEY:
    logger.error("ASSEMBLYAI_API_KEY environment variable not set!")
    sys.exit(1) # Exit if API key is missing
logger.info("Environment variables loaded successfully")

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

# Add CORS middleware - Keep this for initial setup, but we'll also manually add headers
logger.info("Setting up CORS middleware with explicit origins...")
origins = [
    "http://localhost:3000",                  # Your local React app
    "https://typemywordzspeechai.vercel.app", # YOUR NEW, CORRECT LIVE VERCEL URL
    "https://typemywordzspeechai-o03e6tjj3-james-gitukus-projects.vercel.app", # Specific deployment URL
    "https://typemywordzspeechai-*.vercel.app", # All preview deployments for your project
    "https://*.vercel.app",                   # All Vercel deployments (more permissive)
    "*" # Allow all for debugging, but be cautious in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Use the explicit list of origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured successfully")

# Store transcription jobs in memory (for simplicity, reset on server restart)
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
            "language_code": "en_us" # Explicitly set language code
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
async def root(response: Response): # Add Response parameter
    response.headers["Access-Control-Allow-Origin"] = "*" # Explicitly add CORS header
    logger.info("Root endpoint called")
    return {"message": "Transcription Service is running!"}

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks(), response: Response = Response()): # Add Response parameter
    response.headers["Access-Control-Allow-Origin"] = "*" # Explicitly add CORS header
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
        "status": "processing", # Initial status
        "filename": file.filename,
        "created_at": datetime.now().isoformat(),
        "assemblyai_id": None 
    }
    logger.info(f"Job {job_id} initialized with status 'processing'")
    
    # Save the uploaded file temporarily immediately
    try:
        logger.info(f"Saving uploaded file {file.filename} temporarily...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(f"File saved to temporary path: {tmp_path}")
    except Exception as e:
        logger.error(f"ERROR saving temporary file for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save temporary file")

    # Add the heavy processing to a background task
    background_tasks.add_task(process_transcription_job, job_id, tmp_path, file.filename)
    
    # Return immediate response
    logger.info(f"Returning immediate response for job ID: {job_id}")
    return {"job_id": job_id, "status": jobs[job_id]["status"]}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str, response: Response): # Add Response parameter
    response.headers["Access-Control-Allow-Origin"] = "*" # Explicitly add CORS header
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
        
        response = requests.get(transcript_endpoint, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"AssemblyAI status check failed: {response.status_code} - {response.text}")
            job_data.update({
                "status": "failed",
                "error": "Failed to get status from AssemblyAI",
                "completed_at": datetime.now().isoformat()
            })
            return job_data
        
        assemblyai_result = response.json()
        
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
        # If status is still 'queued' or 'processing', do nothing, frontend will poll again
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