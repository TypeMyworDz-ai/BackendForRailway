import logging
import sys
import asyncio
import subprocess
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uuid
from datetime import datetime
from dotenv import load_dotenv
import requests
from pydub import AudioSegment
import stripe
from pydantic import BaseModel
from typing import Optional

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
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

logger.info(f"Attempted to load ASSEMBLYAI_API_KEY. Value found: {bool(ASSEMBLYAI_API_KEY)}")
logger.info(f"Attempted to load STRIPE_SECRET_KEY. Value found: {bool(STRIPE_SECRET_KEY)}")

if not ASSEMBLYAI_API_KEY:
    logger.error("ASSEMBLYAI_API_KEY environment variable not set!")
    sys.exit(1)

if not STRIPE_SECRET_KEY:
    logger.error("STRIPE_SECRET_KEY environment variable not set!")
    sys.exit(1)

# Initialize Stripe
stripe.api_key = STRIPE_SECRET_KEY
logger.info("Stripe initialized successfully")

logger.info("Environment variables loaded successfully")

# Pydantic models for Stripe requests
class CreateSubscriptionRequest(BaseModel):
    priceId: str
    userId: str
    userEmail: str
    userName: str

class UpdateUserPlanRequest(BaseModel):
    userId: str
    planType: str
    subscriptionId: Optional[str] = None

# ENHANCED: Job tracking with better cancellation support
jobs = {}
active_background_tasks = {}  # Track background tasks for cancellation
cancellation_flags = {}  # Track cancellation flags for each job

logger.info("Enhanced job tracking initialized")
# ENHANCED: Ultra aggressive compression function with cancellation checks
def compress_audio_for_transcription(input_path: str, output_path: str = None, job_id: str = None) -> tuple[str, dict]:
    """Compress audio file optimally for AssemblyAI transcription with cancellation support"""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_compressed.mp3"
    
    try:
        # Check for cancellation before starting
        if job_id and cancellation_flags.get(job_id, False):
            logger.info(f"Job {job_id} cancelled during compression setup")
            raise asyncio.CancelledError(f"Job {job_id} was cancelled")
            
        logger.info(f"Compressing {input_path} for transcription...")
        
        # Get original file size
        input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        logger.info(f"Original file size: {input_size:.2f} MB")
        
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        logger.info(f"Original audio: {audio.channels} channels, {audio.frame_rate}Hz, {len(audio)}ms")
        
        # Check for cancellation after loading
        if job_id and cancellation_flags.get(job_id, False):
            logger.info(f"Job {job_id} cancelled during audio loading")
            raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        # ULTRA AGGRESSIVE compression for transcription:
        # 1. Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info("Converted to mono audio")
        
        # 2. Drastically reduce sample rate for speech
        target_sample_rate = 8000  # Even lower - telephone quality
        audio = audio.set_frame_rate(target_sample_rate)
        logger.info(f"Reduced sample rate to {target_sample_rate} Hz")
        
        # Check for cancellation after sample rate conversion
        if job_id and cancellation_flags.get(job_id, False):
            logger.info(f"Job {job_id} cancelled during sample rate conversion")
            raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        # 3. Reduce volume slightly to avoid clipping during compression
        audio = audio - 3  # Reduce by 3dB
        
        # 4. Apply normalization
        audio = audio.normalize()
        logger.info("Applied audio normalization")
        
        # Final cancellation check before export
        if job_id and cancellation_flags.get(job_id, False):
            logger.info(f"Job {job_id} cancelled before export")
            raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        # 5. Export with ULTRA aggressive compression settings
        audio.export(
            output_path, 
            format="mp3",
            bitrate="16k",  # Extremely low bitrate
            parameters=[
                "-q:a", "9",    # Lowest quality = highest compression
                "-ac", "1",     # Force mono
                "-ar", str(target_sample_rate),  # Force very low sample rate
                "-compression_level", "10",  # Maximum compression
                "-joint_stereo", "0",  # Disable joint stereo
                "-reservoir", "0"  # Disable bit reservoir
            ]
        )
        logger.info("Used ultra-aggressive compression settings")
        
        # Calculate compression stats
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            size_difference = input_size - output_size
            if input_size > 0:
                compression_ratio = (size_difference / input_size) * 100
            else:
                compression_ratio = 0
            
            stats = {
                "original_size_mb": round(input_size, 2),
                "compressed_size_mb": round(output_size, 2),
                "compression_ratio_percent": round(compression_ratio, 1),
                "size_reduction_mb": round(size_difference, 2),
                "duration_seconds": len(audio) / 1000.0
            }
            
            logger.info(f"Ultra compression result:")
            logger.info(f"  Original: {stats['original_size_mb']} MB")
            logger.info(f"  Processed: {stats['compressed_size_mb']} MB")
            if size_difference > 0:
                logger.info(f"  Size reduction: {stats['compression_ratio_percent']}% ({stats['size_reduction_mb']} MB saved)")
            else:
                logger.info(f"  Size increase: {abs(stats['compression_ratio_percent'])}% ({abs(stats['size_reduction_mb'])} MB added)")
        
        return output_path, stats
        
    except asyncio.CancelledError:
        logger.info(f"Compression cancelled for job {job_id}")
        # Clean up partial files
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise
        
    except Exception as e:
        logger.error(f"Error compressing audio: {e}")
        # If compression fails, try basic fallback
        try:
            # Check cancellation before fallback
            if job_id and cancellation_flags.get(job_id, False):
                raise asyncio.CancelledError(f"Job {job_id} was cancelled")
                
            audio = AudioSegment.from_file(input_path)
            # Basic fallback compression
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(8000)  # Very low sample rate
            audio.export(output_path, format="mp3", bitrate="16k")
            
            # Recalculate stats for fallback
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            size_difference = input_size - output_size
            compression_ratio = (size_difference / input_size) * 100 if input_size > 0 else 0
            
            stats = {
                "original_size_mb": round(input_size, 2),
                "compressed_size_mb": round(output_size, 2),
                "compression_ratio_percent": round(compression_ratio, 1),
                "size_reduction_mb": round(size_difference, 2),
                "duration_seconds": len(audio) / 1000.0
            }
            
            logger.info("Used fallback compression")
            return output_path, stats
            
        except asyncio.CancelledError:
            logger.info(f"Fallback compression cancelled for job {job_id}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise
            
        except Exception as fallback_error:
            logger.error(f"Fallback compression also failed: {fallback_error}")
            raise

def compress_audio_for_download(input_path: str, output_path: str = None, quality: str = "high") -> str:
    """Compress audio file for download with different quality options"""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_download.mp3"
    
    try:
        logger.info(f"Compressing {input_path} for download (quality: {quality})...")
        
        audio = AudioSegment.from_file(input_path)
        
        # Quality settings - FIXED to ensure compression
        if quality == "high":
            bitrate = "128k"  # Reduced from 192k
            sample_rate = 44100
            channels = 2 if audio.channels > 1 else 1
        elif quality == "medium":
            bitrate = "96k"   # Reduced from 128k
            sample_rate = 22050  # Reduced sample rate
            channels = 1      # Force mono
        else:  # low quality
            bitrate = "64k"
            sample_rate = 16000
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
                "-q:a", "2" if quality == "high" else "5",
                "-ac", str(channels),
                "-ar", str(sample_rate)
            ]
        )
        
        logger.info(f"Download compression complete: {quality} quality")
        return output_path
        
    except Exception as e:
        logger.error(f"Error compressing audio for download: {e}")
        raise
# ENHANCED: Background task with comprehensive cancellation support
async def process_transcription_job(job_id: str, tmp_path: str, filename: str):
    logger.info(f"Background task started for job ID: {job_id}")
    job_data = jobs[job_id]
    
    # Store the task reference for potential cancellation
    active_background_tasks[job_id] = asyncio.current_task()
    # Initialize cancellation flag
    cancellation_flags[job_id] = False

    try:
        # ENHANCED: Multiple cancellation checkpoints with detailed logging
        def check_cancellation():
            if cancellation_flags.get(job_id, False) or job_data.get("status") == "cancelled":
                logger.info(f"Job {job_id} was cancelled - stopping processing")
                raise asyncio.CancelledError(f"Job {job_id} was cancelled")
            return True

        # Check if job was cancelled before processing
        check_cancellation()

        # === Enhanced AssemblyAI Integration with Compression ===
        logger.info(f"Background task: Processing audio {filename} for transcription...")
        
        # Compress audio for optimal transcription with cancellation support
        compressed_path, compression_stats = compress_audio_for_transcription(tmp_path, job_id=job_id)
        
        # Check cancellation again after compression
        check_cancellation()
        
        # Update job with compression stats
        job_data["compression_stats"] = compression_stats
        
        # Upload compressed audio to AssemblyAI using HTTP API
        logger.info("Uploading compressed audio to AssemblyAI...")
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        upload_endpoint = "https://api.assemblyai.com/v2/upload"
        
        # Check cancellation before upload
        check_cancellation()
        
        with open(compressed_path, "rb") as f:
            upload_response = requests.post(upload_endpoint, headers=headers, data=f)
        
        logger.info(f"AssemblyAI upload response status: {upload_response.status_code}")
        logger.info(f"AssemblyAI upload response: {upload_response.text}")
        
        # Check cancellation after upload
        check_cancellation()
        
        if upload_response.status_code != 200:
            logger.error(f"AssemblyAI upload failed: {upload_response.status_code} - {upload_response.text}")
            job_data.update({
                "status": "failed",
                "error": f"Failed to upload audio to AssemblyAI: {upload_response.text}",
                "completed_at": datetime.now().isoformat()
            })
            return
        
        upload_result = upload_response.json()
        audio_url = upload_result["upload_url"]
        logger.info(f"Audio uploaded to AssemblyAI: {audio_url}")

        # Check cancellation before starting transcription
        check_cancellation()

        # Start transcription using HTTP API
        headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
        transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
        json_data = {
            "audio_url": audio_url,
            "language_code": "en_us",
            "punctuate": True,
            "format_text": True
        }
        
        transcript_response = requests.post(transcript_endpoint, headers=headers, json=json_data)
        
        logger.info(f"AssemblyAI transcription start response status: {transcript_response.status_code}")
        logger.info(f"AssemblyAI transcription start response: {transcript_response.text}")
        
        # Final cancellation check before setting AssemblyAI ID
        check_cancellation()
        
        if transcript_response.status_code != 200:
            logger.error(f"AssemblyAI transcription start failed: {transcript_response.status_code} - {transcript_response.text}")
            job_data.update({
                "status": "failed",
                "error": f"Failed to start transcription on AssemblyAI: {transcript_response.text}",
                "completed_at": datetime.now().isoformat()
            })
            return
        
        transcript_result = transcript_response.json()
        transcript_id = transcript_result["id"]
        job_data["assemblyai_id"] = transcript_id
        logger.info(f"AssemblyAI transcription started with ID: {transcript_id}")
        
        # Clean up compressed file
        if os.path.exists(compressed_path):
            os.unlink(compressed_path)
            logger.info(f"Cleaned up compressed file: {compressed_path}")

    except asyncio.CancelledError:
        logger.info(f"Background task for job {job_id} was cancelled")
        # Update job status to cancelled if not already set
        if job_data.get("status") != "cancelled":
            job_data.update({
                "status": "cancelled",
                "cancelled_at": datetime.now().isoformat(),
                "error": "Job was cancelled by user"
            })
        # Clean up any files
        if 'compressed_path' in locals() and os.path.exists(compressed_path):
            os.unlink(compressed_path)
            logger.info(f"Cleaned up compressed file after cancellation: {compressed_path}")
        raise  # Re-raise to properly cancel the task
        
    except Exception as e:
        logger.error(f"Background task: ERROR during transcription for job {job_id}: {str(e)}")
        import traceback
        logger.error(f"Background task: Full traceback: {traceback.format_exc()}")
        job_data.update({
            "status": "failed",
            "error": f"Internal server error during transcription: {str(e)}",
            "completed_at": datetime.now().isoformat()
        })
    finally:
        # Clean up the original temporary file
        if os.path.exists(tmp_path):
            logger.info(f"Background task: Cleaning up original temporary file: {tmp_path}")
            os.unlink(tmp_path)
        
        # Remove from active tasks and cancellation flags
        if job_id in active_background_tasks:
            del active_background_tasks[job_id]
        if job_id in cancellation_flags:
            del cancellation_flags[job_id]
            
        logger.info(f"Background task completed for job ID: {job_id}")

# Background task to monitor application health
async def health_monitor():
    logger.info("Starting health monitor background task")
    while True:
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            logger.info(f"Health Check - Memory: {memory_info.percent}% used, CPU: {cpu_percent}%, Available RAM: {memory_info.available / (1024**3):.2f} GB")
            logger.info(f"Active jobs: {len(jobs)}, Active background tasks: {len(active_background_tasks)}, Cancellation flags: {len(cancellation_flags)}")
            await asyncio.sleep(30)  # Log every 30 seconds
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
            await asyncio.sleep(30)

# NEW: Stripe payment functions
async def create_stripe_customer(email: str, name: str, user_id: str):
    """Create or retrieve Stripe customer"""
    try:
        # Check if customer already exists
        customers = stripe.Customer.list(email=email, limit=1)
        
        if customers.data:
            customer = customers.data[0]
            logger.info(f"Found existing Stripe customer: {customer.id}")
        else:
            # Create new customer
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={"user_id": user_id}
            )
            logger.info(f"Created new Stripe customer: {customer.id}")
        
        return customer
    except Exception as e:
        logger.error(f"Error creating/retrieving Stripe customer: {e}")
        raise

async def create_stripe_subscription(customer_id: str, price_id: str):
    """Create Stripe subscription"""
    try:
        subscription = stripe.Subscription.create(
            customer=customer_id,
            items=[{"price": price_id}],
            payment_behavior="default_incomplete",
            payment_settings={"save_default_payment_method": "on_subscription"},
            expand=["latest_invoice.payment_intent"],
        )
        
        logger.info(f"Created Stripe subscription: {subscription.id}")
        return subscription
    except Exception as e:
        logger.error(f"Error creating Stripe subscription: {e}")
        raise

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
    # Cancel all active background tasks
    for job_id, task in active_background_tasks.items():
        if not task.done():
            logger.info(f"Cancelling background task for job {job_id}")
            cancellation_flags[job_id] = True
            task.cancel()
    # Clear all tracking dictionaries
    jobs.clear()
    active_background_tasks.clear()
    cancellation_flags.clear()
    logger.info("All background tasks cancelled and cleanup complete")
# Create the FastAPI app
logger.info("Creating FastAPI app...")
app = FastAPI(title="Enhanced Transcription Service with Stripe Payments", lifespan=lifespan)
logger.info("FastAPI app created successfully")

# Add CORS middleware with proper configuration
logger.info("Setting up CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured successfully")

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {
        "message": "Enhanced Transcription Service with Stripe Payments is running!",
        "features": [
            "Ultra-aggressive audio compression",
            "Proper job cancellation",
            "Background task management",
            "Real-time status tracking",
            "Stripe payment integration",
            "Subscription management"
        ],
        "stats": {
            "active_jobs": len(jobs),
            "background_tasks": len(active_background_tasks),
            "cancellation_flags": len(cancellation_flags)
        }
    }

# NEW: Stripe payment endpoints
@app.post("/api/create-subscription")
async def create_subscription(request: CreateSubscriptionRequest):
    """Create Stripe subscription for user upgrade"""
    logger.info(f"Creating subscription for user: {request.userEmail}")
    
    try:
        # Create or get Stripe customer
        customer = await create_stripe_customer(
            email=request.userEmail,
            name=request.userName,
            user_id=request.userId
        )
        
        # Create subscription
        subscription = await create_stripe_subscription(
            customer_id=customer.id,
            price_id=request.priceId
        )
        
        # Return client secret for payment confirmation
        return {
            "subscriptionId": subscription.id,
            "clientSecret": subscription.latest_invoice.payment_intent.client_secret,
            "customerId": customer.id
        }
        
    except Exception as e:
        logger.error(f"Error creating subscription: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to create subscription: {str(e)}")

@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events"""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        if STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        else:
            # For testing without webhook secret
            event = stripe.Event.construct_from(
                json.loads(payload), stripe.api_key
            )
        
        logger.info(f"Received Stripe webhook: {event['type']}")
        
        # Handle successful payment
        if event['type'] == 'invoice.payment_succeeded':
            invoice = event['data']['object']
            customer_id = invoice['customer']
            subscription_id = invoice['subscription']
            
            # Get customer details
            customer = stripe.Customer.retrieve(customer_id)
            user_id = customer.metadata.get('user_id')
            
            logger.info(f"Payment succeeded for user {user_id}, subscription {subscription_id}")
            
            # Here you would update your user's plan in Firebase
            # This would typically involve calling your Firebase Admin SDK
            # For now, we'll just log it
            logger.info(f"User {user_id} payment successful - plan should be upgraded")
            
        elif event['type'] == 'invoice.payment_failed':
            invoice = event['data']['object']
            customer_id = invoice['customer']
            
            customer = stripe.Customer.retrieve(customer_id)
            user_id = customer.metadata.get('user_id')
            
            logger.warning(f"Payment failed for user {user_id}")
            
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            customer_id = subscription['customer']
            
            customer = stripe.Customer.retrieve(customer_id)
            user_id = customer.metadata.get('user_id')
            
            logger.info(f"Subscription cancelled for user {user_id}")
            
        return {"status": "success"}
        
    except ValueError as e:
        logger.error(f"Invalid payload in webhook: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature in webhook: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")

@app.post("/api/cancel-subscription")
async def cancel_subscription(user_id: str):
    """Cancel user's Stripe subscription"""
    try:
        # Find customer by user_id metadata
        customers = stripe.Customer.list(limit=100)
        customer = None
        
        for c in customers.auto_paging_iter():
            if c.metadata.get('user_id') == user_id:
                customer = c
                break
                
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
            
        # Get active subscriptions
        subscriptions = stripe.Subscription.list(customer=customer.id, status='active')
        
        if not subscriptions.data:
            raise HTTPException(status_code=404, detail="No active subscription found")
            
        # Cancel the first active subscription
        subscription = subscriptions.data[0]
        cancelled_subscription = stripe.Subscription.delete(subscription.id)
        
        logger.info(f"Cancelled subscription {subscription.id} for user {user_id}")
        
        return {
            "message": "Subscription cancelled successfully",
            "subscription_id": cancelled_subscription.id
        }
        
    except Exception as e:
        logger.error(f"Error cancelling subscription: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to cancel subscription: {str(e)}")

@app.get("/api/subscription-status/{user_id}")
async def get_subscription_status(user_id: str):
    """Get user's current subscription status"""
    try:
        # Find customer by user_id metadata
        customers = stripe.Customer.list(limit=100)
        customer = None
        
        for c in customers.auto_paging_iter():
            if c.metadata.get('user_id') == user_id:
                customer = c
                break
                
        if not customer:
            return {"has_subscription": False, "plan": "free"}
            
        # Get active subscriptions
        subscriptions = stripe.Subscription.list(customer=customer.id, status='active')
        
        if not subscriptions.data:
            return {"has_subscription": False, "plan": "free"}
            
        subscription = subscriptions.data[0]
        price_id = subscription.items.data[0].price.id
        
        # Map price_id to plan name (you'll need to customize this)
        plan_mapping = {
            "price_1S8xVnLgugZakECYNFDOMVwh": "pro",  # Your actual price ID
            # Add more mappings as needed
        }
        
        plan = plan_mapping.get(price_id, "unknown")
        
        return {
            "has_subscription": True,
            "plan": plan,
            "subscription_id": subscription.id,
            "status": subscription.status,
            "current_period_end": subscription.current_period_end
        }
        
    except Exception as e:
        logger.error(f"Error getting subscription status: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to get subscription status: {str(e)}")
@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    logger.info(f"Transcribe endpoint called with file: {file.filename}")
    
    # Check if file is audio or video
    if not file.content_type.startswith(('audio/', 'video/')):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Please upload an audio or video file")
    
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"Created job ID: {job_id}")
    
    # Initialize job status with enhanced tracking
    jobs[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "created_at": datetime.now().isoformat(),
        "assemblyai_id": None,
        "compression_stats": None,
        "file_size_mb": 0,
        "content_type": file.content_type
    }
    
    # Initialize cancellation flag
    cancellation_flags[job_id] = False
    logger.info(f"Job {job_id} initialized with status 'processing'")
    
    # Save the uploaded file temporarily
    try:
        logger.info(f"Saving uploaded file {file.filename} temporarily...")
        # Save original file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Calculate and store file size
        file_size_mb = len(content) / (1024 * 1024)
        jobs[job_id]["file_size_mb"] = round(file_size_mb, 2)
        
        logger.info(f"File saved to: {tmp_path} (Size: {file_size_mb:.2f} MB)")
        
    except Exception as e:
        logger.error(f"ERROR processing file for job {job_id}: {str(e)}")
        # Clean up job tracking
        if job_id in jobs:
            del jobs[job_id]
        if job_id in cancellation_flags:
            del cancellation_flags[job_id]
        raise HTTPException(status_code=500, detail="Failed to process audio file")

    # Add the processing to background task
    background_tasks.add_task(process_transcription_job, job_id, tmp_path, file.filename)
    
    # Return immediate response with enhanced info
    logger.info(f"Returning immediate response for job ID: {job_id}")
    return {
        "job_id": job_id, 
        "status": jobs[job_id]["status"],
        "filename": file.filename,
        "file_size_mb": jobs[job_id]["file_size_mb"],
        "created_at": jobs[job_id]["created_at"]
    }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    logger.info(f"Status check for job ID: {job_id}")
    if job_id not in jobs:
        logger.warning(f"Job ID {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs[job_id]
    
    # If job is cancelled, return cancelled status immediately
    if job_data["status"] == "cancelled" or cancellation_flags.get(job_id, False):
        logger.info(f"Job {job_id} was cancelled, returning cancelled status")
        job_data["status"] = "cancelled"  # Ensure status is set correctly
        return job_data
    
    # If transcription is still processing, poll AssemblyAI for status
    if job_data["status"] == "processing" and job_data["assemblyai_id"]:
        logger.info(f"Polling AssemblyAI for status of transcript ID: {job_data['assemblyai_id']}")
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        transcript_endpoint = f"https://api.assemblyai.com/v2/transcript/{job_data['assemblyai_id']}"
        
        try:
            # Check for cancellation before making API call
            if cancellation_flags.get(job_id, False):
                logger.info(f"Job {job_id} was cancelled during status check")
                job_data.update({
                    "status": "cancelled",
                    "cancelled_at": datetime.now().isoformat(),
                    "error": "Job was cancelled by user"
                })
                return job_data
            
            response_data = requests.get(transcript_endpoint, headers=headers)
            
            if response_data.status_code != 200:
                logger.error(f"AssemblyAI status check failed: {response_data.status_code} - {response_data.text}")
                job_data.update({
                    "status": "failed",
                    "error": f"Failed to get status from AssemblyAI: {response_data.text}",
                    "completed_at": datetime.now().isoformat()
                })
                return job_data
            
            assemblyai_result = response_data.json()
            
            # Check if job was cancelled while AssemblyAI was processing
            if cancellation_flags.get(job_id, False) or job_data["status"] == "cancelled":
                logger.info(f"Job {job_id} was cancelled during AssemblyAI processing")
                job_data.update({
                    "status": "cancelled",
                    "cancelled_at": datetime.now().isoformat(),
                    "error": "Job was cancelled by user"
                })
                return job_data
            
            if assemblyai_result["status"] == "completed":
                logger.info(f"AssemblyAI transcription {job_data['assemblyai_id']} completed.")
                job_data.update({
                    "status": "completed",
                    "transcription": assemblyai_result["text"],
                    "language": assemblyai_result["language_code"],
                    "completed_at": datetime.now().isoformat(),
                    "word_count": len(assemblyai_result["text"].split()) if assemblyai_result["text"] else 0,
                    "duration_seconds": assemblyai_result.get("audio_duration", 0)
                })
            elif assemblyai_result["status"] == "error":
                logger.error(f"AssemblyAI transcription {job_data['assemblyai_id']} failed: {assemblyai_result.get('error', 'Unknown error')}")
                job_data.update({
                    "status": "failed",
                    "error": assemblyai_result.get("error", "Transcription failed on AssemblyAI"),
                    "completed_at": datetime.now().isoformat()
                })
            else:
                logger.info(f"AssemblyAI transcription {job_data['assemblyai_id']} status: {assemblyai_result['status']}")
                # Update job with current AssemblyAI status for better tracking
                job_data["assemblyai_status"] = assemblyai_result["status"]
        
        except Exception as e:
            logger.error(f"Error polling AssemblyAI status: {str(e)}")
            job_data.update({
                "status": "failed",
                "error": f"Error checking transcription status: {str(e)}",
                "completed_at": datetime.now().isoformat()
            })

    return job_data

@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Enhanced cancel endpoint with comprehensive job termination"""
    logger.info(f"Cancel request received for job ID: {job_id}")
    
    if job_id not in jobs:
        logger.warning(f"Cancel request: Job ID {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs[job_id]
    
    try:
        # Set cancellation flag immediately
        cancellation_flags[job_id] = True
        logger.info(f"Cancellation flag set for job {job_id}")
        
        # Cancel the background task if it's still running
        if job_id in active_background_tasks:
            task = active_background_tasks[job_id]
            if not task.done():
                logger.info(f"Cancelling active background task for job {job_id}")
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)  # Wait up to 2 seconds for graceful cancellation
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.info(f"Background task for job {job_id} cancelled (timeout/cancelled)")
            else:
                logger.info(f"Background task for job {job_id} was already completed")
        
        # Update job status to cancelled
        job_data.update({
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat(),
            "error": "Job was cancelled by user"
        })
        
        # Note about AssemblyAI: We can't actually cancel jobs on AssemblyAI's end
        # but we mark them as cancelled in our system and ignore the results
        if job_data.get("assemblyai_id"):
            logger.info(f"AssemblyAI job {job_data['assemblyai_id']} cannot be cancelled on their end, but marked as cancelled in our system")
            job_data["assemblyai_note"] = "AssemblyAI job continues but results will be ignored"
        
        logger.info(f"Job {job_id} successfully cancelled")
        return {
            "message": "Job cancelled successfully", 
            "job_id": job_id,
            "cancelled_at": job_data["cancelled_at"],
            "previous_status": job_data.get("previous_status", "processing")
        }
        
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}")
        # Even if there's an error, mark the job as cancelled
        job_data.update({
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat(),
            "error": f"Job cancelled with errors: {str(e)}"
        })
        raise HTTPException(status_code=500, detail=f"Job cancelled but with errors: {str(e)}")
@app.post("/compress-download")
async def compress_download(file: UploadFile = File(...), quality: str = "high"):
    """Endpoint to compress audio files for download"""
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

@app.delete("/cleanup")
async def cleanup_old_jobs():
    """Enhanced cleanup endpoint with better job management"""
    logger.info("Cleanup endpoint called")
    
    current_time = datetime.now()
    jobs_to_remove = []
    tasks_to_cancel = []
    flags_to_remove = []
    
    for job_id, job_data in jobs.items():
        # Remove jobs older than 1 hour
        created_at = datetime.fromisoformat(job_data["created_at"])
        age_hours = (current_time - created_at).total_seconds() / 3600
        
        if age_hours > 1 and job_data["status"] in ["completed", "failed", "cancelled"]:
            jobs_to_remove.append(job_id)
            
            # Also clean up related tracking
            if job_id in active_background_tasks:
                task = active_background_tasks[job_id]
                if not task.done():
                    tasks_to_cancel.append((job_id, task))
                    
            if job_id in cancellation_flags:
                flags_to_remove.append(job_id)
    
    # Cancel old background tasks
    for job_id, task in tasks_to_cancel:
        try:
            task.cancel()
            logger.info(f"Cancelled old background task for job: {job_id}")
        except Exception as e:
            logger.error(f"Error cancelling old task {job_id}: {e}")
    
    # Remove old jobs and tracking data
    for job_id in jobs_to_remove:
        del jobs[job_id]
        logger.info(f"Cleaned up old job: {job_id}")
        
    for job_id in flags_to_remove:
        if job_id in active_background_tasks:
            del active_background_tasks[job_id]
        if job_id in cancellation_flags:
            del cancellation_flags[job_id]
    
    cleanup_stats = {
        "jobs_removed": len(jobs_to_remove),
        "tasks_cancelled": len(tasks_to_cancel),
        "flags_cleared": len(flags_to_remove),
        "remaining_jobs": len(jobs),
        "active_tasks": len(active_background_tasks),
        "active_flags": len(cancellation_flags)
    }
    
    logger.info(f"Cleanup completed: {cleanup_stats}")
    return {
        "message": f"Cleaned up {len(jobs_to_remove)} old jobs",
        "stats": cleanup_stats
    }

@app.get("/jobs")
async def list_jobs():
    """Enhanced jobs list endpoint with better information"""
    logger.info("Jobs list endpoint called")
    
    job_summary = {}
    for job_id, job_data in jobs.items():
        job_summary[job_id] = {
            "status": job_data["status"],
            "filename": job_data.get("filename", "unknown"),
            "created_at": job_data["created_at"],
            "file_size_mb": job_data.get("file_size_mb", 0),
            "assemblyai_id": job_data.get("assemblyai_id"),
            "assemblyai_status": job_data.get("assemblyai_status"),
            "has_background_task": job_id in active_background_tasks,
            "is_cancellation_flagged": cancellation_flags.get(job_id, False),
            "word_count": job_data.get("word_count"),
            "duration_seconds": job_data.get("duration_seconds")
        }
    
    return {
        "total_jobs": len(jobs),
        "active_background_tasks": len(active_background_tasks),
        "cancellation_flags": len(cancellation_flags),
        "jobs": job_summary,
        "system_stats": {
            "jobs_by_status": {
                status: len([j for j in jobs.values() if j["status"] == status])
                for status in ["processing", "completed", "failed", "cancelled"]
            }
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    logger.info("Health check endpoint called")
    
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "memory_percent": memory_info.percent,
                "cpu_percent": cpu_percent,
                "available_ram_gb": round(memory_info.available / (1024**3), 2)
            },
            "application": {
                "total_jobs": len(jobs),
                "active_background_tasks": len(active_background_tasks),
                "cancellation_flags": len(cancellation_flags),
                "jobs_by_status": {
                    status: len([j for j in jobs.values() if j["status"] == status])
                    for status in ["processing", "completed", "failed", "cancelled"]
                }
            }
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

logger.info("=== FASTAPI APPLICATION SETUP COMPLETE ===")

# Final validation and startup logging
logger.info("Performing final system validation...")
logger.info(f"AssemblyAI API Key configured: {bool(ASSEMBLYAI_API_KEY)}")
logger.info(f"Stripe Secret Key configured: {bool(STRIPE_SECRET_KEY)}")
logger.info(f"Job tracking systems initialized:")
logger.info(f"  - Main jobs dictionary: {len(jobs)} jobs")
logger.info(f"  - Active background tasks: {len(active_background_tasks)} tasks")
logger.info(f"  - Cancellation flags: {len(cancellation_flags)} flags")

# Log all available endpoints
logger.info("Available API endpoints:")
logger.info("  POST /transcribe - Start new transcription job")
logger.info("  GET /status/{job_id} - Check job status")
logger.info("  POST /cancel/{job_id} - Cancel transcription job")
logger.info("  POST /compress-download - Compress audio for download")
logger.info("  POST /api/create-subscription - Create Stripe subscription")
logger.info("  POST /api/stripe-webhook - Handle Stripe webhooks")
logger.info("  POST /api/cancel-subscription - Cancel user subscription")
logger.info("  GET /api/subscription-status/{user_id} - Get subscription status")
logger.info("  GET /jobs - List all jobs")
logger.info("  GET /health - System health check")
logger.info("  DELETE /cleanup - Clean up old jobs")
logger.info("  GET / - Root endpoint with service info")

# Run this if the file is executed directly
if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly...")
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting enhanced transcription service with Stripe payments on {host}:{port}")
    logger.info("🚀 ENHANCED FEATURES ENABLED:")
    logger.info("  ✅ Ultra-aggressive audio compression (16k bitrate, 8kHz)")
    logger.info("  ✅ Comprehensive job cancellation system")
    logger.info("  ✅ Background task management with proper cleanup")
    logger.info("  ✅ Real-time status tracking with AssemblyAI polling")
    logger.info("  ✅ Enhanced error handling and recovery")
    logger.info("  ✅ System health monitoring every 30 seconds")
    logger.info("  ✅ Automatic cleanup of old jobs (1+ hours)")
    logger.info("  ✅ Detailed logging and debugging support")
    logger.info("  ✅ CORS enabled for frontend integration")
    logger.info("  ✅ Multiple cancellation checkpoints during processing")
    logger.info("  ✅ Stripe payment integration for subscriptions")
    logger.info("  ✅ Webhook handling for payment events")
    logger.info("  ✅ Subscription management endpoints")
    
    logger.info("🔧 TECHNICAL IMPROVEMENTS:")
    logger.info("  - Cancellation flags prevent race conditions")
    logger.info("  - Background tasks are properly tracked and cancelled")
    logger.info("  - File cleanup happens even on cancellation")
    logger.info("  - AssemblyAI jobs continue but results are ignored when cancelled")
    logger.info("  - Enhanced status endpoint with detailed job information")
    logger.info("  - Comprehensive health monitoring and system stats")
    logger.info("  - Stripe customer and subscription management")
    logger.info("  - Secure webhook signature verification")
    
    try:
        uvicorn.run(
            app, 
            host=host, 
            port=port,
            log_level="info",
            access_log=True,
            reload=False,  # Disable reload in production
            workers=1      # Single worker to maintain job state consistency
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
else:
    logger.info("Application loaded as module")
    logger.info("Ready to handle requests with enhanced job cancellation and Stripe payment support")