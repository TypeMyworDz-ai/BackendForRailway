import logging
import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

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

# Load environment variables (like OPENAI_API_KEY)
logger.info("Loading environment variables...")
load_dotenv()
logger.info("Environment variables loaded successfully")

# Create the FastAPI app
logger.info("Creating FastAPI app...")
app = FastAPI(title="Transcription Service")
logger.info("FastAPI app created successfully")

# Add CORS middleware
logger.info("Setting up CORS middleware...")
origins = [
    "http://localhost:3000",                  # Your local React app
    "https://typemywordzaiapp-git-main-james-gitukus-projects.vercel.app",    # Your Vercel preview frontend URL (if still active)
    "https://typemywordzspeechai.vercel.app", # YOUR NEW, CORRECT LIVE VERCEL URL
    # Add any other frontend URLs that need to access this backend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured successfully")

# Global variable to store the model
model = None

async def get_whisper_model():
    global model
    logger.info("get_whisper_model() called")
    if model is None:
        try:
            logger.info("Model is None, attempting to load Whisper model...")
            logger.info("Loading Whisper model 'tiny.en'... This might take a moment.")
            
            # Check available memory before loading
            import psutil
            memory_info = psutil.virtual_memory()
            logger.info(f"Available memory before model load: {memory_info.available / (1024**3):.2f} GB")
            logger.info(f"Memory usage percentage: {memory_info.percent}%")
            
            model = whisper.load_model("tiny.en")
            logger.info("Whisper model loaded successfully!")
            
            # Check memory after loading
            memory_info_after = psutil.virtual_memory()
            logger.info(f"Available memory after model load: {memory_info_after.available / (1024**3):.2f} GB")
            logger.info(f"Memory usage percentage after load: {memory_info_after.percent}%")
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR loading Whisper model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e
    else:
        logger.info("Model already loaded, returning existing model")
    return model

# Store transcription jobs in memory (for simplicity, reset on server restart)
jobs = {}
logger.info("Jobs dictionary initialized")

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "Transcription Service is running!"}

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
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
        "created_at": datetime.now().isoformat()
    }
    logger.info(f"Job {job_id} initialized with status 'processing'")
    
    try:
        # Save the uploaded file temporarily
        logger.info(f"Saving uploaded file {file.filename} temporarily...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(f"File saved to temporary path: {tmp_path}")
        
        # Transcribe the audio/video
        logger.info(f"Starting transcription for {file.filename}")
        # Get the model (loads it if not already loaded)
        logger.info("Calling get_whisper_model()...")
        whisper_model = await get_whisper_model()
        logger.info("Whisper model obtained, starting transcription...")
        
        result = whisper_model.transcribe(tmp_path)
        logger.info(f"Transcription completed for {file.filename}")
        
        # Clean up the temporary file
        logger.info(f"Cleaning up temporary file: {tmp_path}")
        os.unlink(tmp_path)
        
        # Update job with results
        jobs[job_id].update({
            "status": "completed",
            "transcription": result["text"],
            "language": result["language"],
            "completed_at": datetime.now().isoformat()
        })
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        # If something goes wrong
        logger.error(f"ERROR in transcription for job {job_id}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        logger.info(f"Job {job_id} marked as failed")
    
    return {"job_id": job_id, "status": jobs[job_id]["status"]}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    logger.info(f"Status check for job ID: {job_id}")
    if job_id not in jobs:
        logger.warning(f"Job ID {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

logger.info("=== FASTAPI APPLICATION SETUP COMPLETE ===")

# Run this if the file is executed directly
if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly...")
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)