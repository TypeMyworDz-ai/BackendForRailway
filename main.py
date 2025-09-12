from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import whisper
import tempfile
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import logging

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(title="Transcription Service")

# Add CORS middleware
# IMPORTANT: Adjust origins for your specific frontend deployment URLs
origins = [
    "http://localhost:3000",                  # Your local React app
    "https://typemywordzaiapp-git-main-james-gitukus-projects.vercel.app",    # Your Vercel preview frontend URL
    "https://typemywordzspeechai.vercel.app", # Your live Vercel URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the model
model = None

async def get_whisper_model():
    global model
    if model is None:
        logger.info("Loading Whisper model... This might take a moment.")
        # Using the 'tiny.en' model for English-only and less memory
        model = whisper.load_model("tiny.en")
        logger.info("Model loaded successfully!")
    return model

# Store transcription jobs in memory (for simplicity, reset on server restart)
jobs = {}

@app.get("/")
async def root():
    return {"message": "Transcription Service is running!"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/favicon.ico")
async def favicon():
    # If you have a favicon.ico in static/, serve it
    # Otherwise, return 204 No Content to avoid 404
    favicon_path = "static/favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return Response(status_code=204)

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    # Check if file is audio or video
    if not file.content_type.startswith(('audio/', 'video/')):
        raise HTTPException(status_code=400, detail="Please upload an audio or video file")
    
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    jobs[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Transcribe the audio/video
        logger.info(f"Starting transcription for {file.filename}")
        # Get the model (loads it if not already loaded)
        whisper_model = await get_whisper_model()
        result = whisper_model.transcribe(tmp_path)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        # Update job with results
        jobs[job_id].update({
            "status": "completed",
            "transcription": result["text"],
            "language": result["language"],
            "completed_at": datetime.now().isoformat()
        })
        
        logger.info(f"Transcription completed for {file.filename}")
        
    except Exception as e:
        # If something goes wrong
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        logger.error(f"Transcription failed: {str(e)}")
    
    return {"job_id": job_id, "status": jobs[job_id]["status"]}

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

# Run this if the file is executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)