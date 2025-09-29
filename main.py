# ===============================================================================
# COMPLETE UPDATED main.py - PART 1 (WITH NEW AI ACCESS LOGIC)
# This file includes Anthropic Claude integration for user-facing and admin-facing AI features.
# ===============================================================================

import logging
import sys
import asyncio
import subprocess
import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Response, Request, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uuid
from datetime import datetime
from dotenv import load_dotenv
import requests
from pydub import AudioSegment
from pydantic import BaseModel
from typing import Optional, List
import httpx
# NEW IMPORTS for python-docx and regex
from docx import Document
from docx.shared import Inches
from io import BytesIO
from fastapi.responses import StreamingResponse
import re
import anthropic

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

# Define codenames for services (UPDATED)
TYPEMYWORDZ1_NAME = "TypeMyworDz1"
TYPEMYWORDZ2_NAME = "TypeMyworDz2"
TYPEMYWORDZ_AI_NAME = "TypeMyworDz AI"

# Install ffmpeg if not available
def install_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        logger.info("ffmpeg is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("Installing ffmpeg... (This might not be strictly necessary if pydub uses a pre-installed one on Railway)")
        try:
            subprocess.run(['apt-get', 'update'], check=True)
            subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
            logger.info("ffmpeg installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install ffmpeg: {e}")

install_ffmpeg()
logger.info("Loading environment variables...")
load_dotenv()
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
RENDER_WHISPER_URL = os.getenv("RENDER_WHISPER_URL")

# ADDED: Load Anthropic API Key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")
PAYSTACK_PUBLIC_KEY = os.getenv("PAYSTACK_PUBLIC_KEY")
PAYSTACK_WEBHOOK_SECRET = os.getenv("PAYSTACK_WEBHOOK_SECRET")

logger.info(f"Attempted to load ASSEMBLYAI_API_KEY. Value found: {bool(ASSEMBLYAI_API_KEY)}")
logger.info(f"Attempted to load RENDER_WHISPER_URL. Value found: {bool(RENDER_WHISPER_URL)}")
logger.info(f"Attempted to load ANTHROPIC_API_KEY. Value found: {bool(ANTHROPIC_API_KEY)}")
logger.info(f"Attempted to load PAYSTACK_SECRET_KEY. Value found: {bool(PAYSTACK_SECRET_KEY)}")
logger.info(f"Attempted to load PAYSTACK_PUBLIC_KEY. Value found: {bool(PAYSTACK_PUBLIC_KEY)}")

if not ASSEMBLYAI_API_KEY:
    logger.error(f"{TYPEMYWORDZ1_NAME} API Key environment variable not set! {TYPEMYWORDZ1_NAME} will not work as primary or fallback.")

if not RENDER_WHISPER_URL:
    logger.warning(f"{TYPEMYWORDZ2_NAME} URL environment variable not set! {TYPEMYWORDZ2_NAME} will not be available as a fallback.")

# IMPROVED: Check for Anthropic API Key with validation
if not ANTHROPIC_API_KEY:
    logger.warning(f"{TYPEMYWORDZ_AI_NAME} API Key environment variable not set! {TYPEMYWORDZ_AI_NAME} features will be disabled.")
elif not ANTHROPIC_API_KEY.startswith('sk-ant-'):
    logger.warning(f"{TYPEMYWORDZ_AI_NAME} API key format appears invalid (should start with 'sk-ant-')")

if not PAYSTACK_SECRET_KEY:
    logger.warning("PAYSTACK_SECRET_KEY environment variable not set! Paystack features will be disabled.")

if PAYSTACK_SECRET_KEY:
    logger.info("Paystack configuration found - payment verification enabled")
else:
    logger.warning("Paystack configuration missing - payment verification disabled")

logger.info("Environment variables loaded successfully")

# IMPROVED: Anthropic Client Initialization with better error handling
claude_client = None
if ANTHROPIC_API_KEY:
    try:
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        # Test the connection with a minimal request to validate the key
        logger.info(f"{TYPEMYWORDZ_AI_NAME} client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing {TYPEMYWORDZ_AI_NAME} client: {e}")
        claude_client = None
else:
    logger.warning(f"{TYPEMYWORDZ_AI_NAME} API key is missing, Claude client will not be initialized.")

# NEW: Helper function to determine if a user has access to AI features
def is_paid_ai_user(user_plan: str) -> bool:
    # UPDATED: Only 'Three-Day Plan' and 'Pro' plans are eligible for AI features
    paid_plans_for_ai = ['Three-Day Plan', 'Pro']
    return user_plan in paid_plans_for_ai


# Pydantic Models
class PaystackVerificationRequest(BaseModel):
    reference: str

class PaystackInitializationRequest(BaseModel):
    email: str
    amount: float
    plan_name: str
    user_id: str
    country_code: str
    callback_url: str

class PaystackWebhookRequest(BaseModel):
    event: str
    data: dict

class CreditUpdateRequest(BaseModel):
    email: str
    plan_name: str
    amount: float
    currency: str
    duration_hours: Optional[int] = None
    duration_days: Optional[int] = None

class FormattedWordDownloadRequest(BaseModel):
    transcription_html: str
    filename: Optional[str] = "transcription.docx"

# FIXED: Updated Pydantic Models for AI Interaction with most likely working model
class UserAIRequest(BaseModel):
    transcript: str
    user_prompt: str
    model: str = "claude-3-haiku-20240307"  # UPDATED to working model
    max_tokens: int = 1000

class AdminAIFormatRequest(BaseModel):
    transcript: str
    formatting_instructions: str = "Format the transcript for readability, correct grammar, and identify main sections with headings. Ensure a professional tone."
    model: str = "claude-3-5-haiku-20241022"  # UPDATED to latest working model
    max_tokens: int = 4000

# Global variables for job tracking
jobs = {}
active_background_tasks = {}
cancellation_flags = {}

logger.info("Enhanced job tracking initialized")
# ===============================================================================
# main.py - Part 2 of 7: Audio Processing and Helper Functions (UNCHANGED)
# ===============================================================================

# Audio analysis and compression functions
async def analyze_audio_characteristics(audio_path: str) -> dict:
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_seconds = len(audio) / 1000.0
        
        if audio.dBFS < -50:
            quality_score = 0.1
        elif audio.dBFS < -30:
            quality_score = 0.4
        else:
            quality_score = 0.8
            
        language = "unknown"
        return {
            "duration_seconds": duration_seconds,
            "quality_score": quality_score,
            "language": language,
            "channels": audio.channels,
            "sample_rate": audio.frame_rate,
            "size_mb": os.path.getsize(audio_path) / (1024 * 1024)
        }
    except Exception as e:
        logger.error(f"Error analyzing audio characteristics: {e}")
        return {
            "duration_seconds": 0,
            "quality_score": 0,
            "language": "unknown",
            "channels": 0,
            "sample_rate": 0,
            "size_mb": 0,
            "error": str(e)
        }

def compress_audio_for_transcription(input_path: str, output_path: str = None, job_id: str = None) -> tuple[str, dict]:
    """Compress audio file optimally for transcription with cancellation support"""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_compressed.mp3"
    
    try:
        if job_id and cancellation_flags.get(job_id, False):
            logger.info(f"Job {job_id} cancelled during compression setup")
            raise asyncio.CancelledError(f"Job {job_id} was cancelled")
            
        logger.info(f"Compressing {input_path} for transcription...")
        
        input_size = os.path.getsize(input_path) / (1024 * 1024)
        logger.info(f"Original file size: {input_size:.2f} MB")
        
        audio = AudioSegment.from_file(input_path)
        logger.info(f"Original audio: {audio.channels} channels, {audio.frame_rate}Hz, {len(audio)}ms")
        
        if job_id and cancellation_flags.get(job_id, False):
            logger.info(f"Job {job_id} cancelled during audio loading")
            raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info("Converted to mono audio")
        
        target_sample_rate = 16000
        audio = audio.set_frame_rate(target_sample_rate)
        logger.info(f"Reduced sample rate to {target_sample_rate} Hz")
        
        if job_id and cancellation_flags.get(job_id, False):
            logger.info(f"Job {job_id} cancelled during sample rate conversion")
            raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        audio = audio - 3
        audio = audio.normalize()
        logger.info("Applied audio normalization")
        
        if job_id and cancellation_flags.get(job_id, False):
            logger.info(f"Job {job_id} cancelled before export")
            raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        audio.export(
            output_path, 
            format="mp3",
            bitrate="64k",
            parameters=[
                "-q:a", "9",
                "-ac", "1",
                "-ar", str(target_sample_rate)
            ]
        )
        logger.info("Audio compression complete")
        
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            
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
            
            logger.info(f"Compression result:")
            logger.info(f"  Original: {stats['original_size_mb']} MB")
            logger.info(f"  Processed: {stats['compressed_size_mb']} MB")
            if size_difference > 0:
                logger.info(f"  Size reduction: {stats['compression_ratio_percent']}% ({stats['size_reduction_mb']} MB saved)")
            else:
                logger.info(f"  Size increase: {abs(stats['compression_ratio_percent'])}% ({abs(stats['size_reduction_mb'])} MB added)")
        
        return output_path, stats
        
    except asyncio.CancelledError:
        logger.info(f"Compression cancelled for job {job_id}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise
        
    except Exception as e:
        logger.error(f"Error compressing audio: {e}")
        try:
            if job_id and cancellation_flags.get(job_id, False):
                raise asyncio.CancelledError(f"Job {job_id} was cancelled")
                
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            audio.export(output_path, format="mp3", bitrate="64k")
            
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            size_difference = input_size - output_size
            compression_ratio = (size_difference / input_path) * 100 if input_path > 0 else 0
            
            stats = {
                "original_size_mb": round(input_size, 2),
                "compressed_size_mb": round(output_path, 2),
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
        
        if quality == "high":
            bitrate = "128k"
            sample_rate = 44100
            channels = 2 if audio.channels > 1 else 1
        elif quality == "medium":
            bitrate = "96k"
            sample_rate = 22050
            channels = 1
        else:
            bitrate = "64k"
            sample_rate = 16000
            channels = 1
        
        if audio.channels != channels:
            audio = audio.set_channels(channels)
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)
        
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

# Currency Conversion and Channel Mapping Logic
USD_TO_LOCAL_RATES = {
    'KE': 145.0,
    'NG': 1500.0,
    'GH': 15.0,
    'ZA': 19.0,
    'OTHER_AFRICA': 'USD',
}

COUNTRY_CURRENCY_MAP = {
    'KE': 'KES',
    'NG': 'NGN',
    'GH': 'GHS',
    'ZA': 'ZAR',
    'OTHER_AFRICA': 'USD',
}

COUNTRY_CHANNELS_MAP = {
    'KE': ['mobile_money', 'card'],
    'NG': ['bank', 'ussd', 'mobile_money', 'card'],
    'GH': ['mobile_money', 'card'],
    'ZA': ['eft', 'card'],
    'OTHER_AFRICA': ['card'],
}

def get_local_amount_and_currency(base_usd_amount: float, country_code: str) -> tuple[float, str]:
    currency = COUNTRY_CURRENCY_MAP.get(country_code, 'USD')
    if currency == 'USD':
        return base_usd_amount, 'USD'
    
    rate = USD_TO_LOCAL_RATES.get(country_code, 1.0)
    local_amount = round(base_usd_amount * rate, 2)
    return local_amount, currency

def get_payment_channels(country_code: str) -> list[str]:
    return COUNTRY_CHANNELS_MAP.get(country_code, ['card'])

# Health monitor function
async def health_monitor():
    logger.info("Starting health monitor background task")
    while True:
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            logger.info(f"Health Check - Memory: {memory_info.percent}% used, CPU: {cpu_percent}%, Available RAM: {memory_info.available / (1024**3):.2f} GB")
            logger.info(f"Active jobs: {len(jobs)}, Active background tasks: {len(active_background_tasks)}, Cancellation flags: {len(cancellation_flags)}")
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
            await asyncio.sleep(30)
# ===============================================================================
# main.py - Part 3 of 7: Paystack Functions (UNCHANGED)
# ===============================================================================

# Paystack helper functions
async def verify_paystack_payment(reference: str) -> dict:
    """Verify Paystack payment using reference"""
    if not PAYSTACK_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Paystack configuration missing")
    
    headers = {
        'Authorization': f'Bearer {PAYSTACK_SECRET_KEY}',
        'Content-Type': 'application/json'
    }
    
    try:
        logger.info(f"Verifying Paystack payment with reference: {reference}")
        response = requests.get(
            f'https://api.paystack.co/transaction/verify/{reference}',
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            payment_data = response.json()
            
            if payment_data['status'] and payment_data['data']['status'] == 'success':
                amount_kobo = payment_data['data']['amount']
                amount = amount_kobo / 100
                customer_email = payment_data['data']['customer']['email']
                currency = payment_data['data']['currency']
                plan_name = payment_data['data']['metadata'].get('plan', 'Unknown')
                
                logger.info(f"✅ Paystack payment verified: {customer_email} paid {amount} {currency} for {plan_name}")
                
                return {
                    'status': 'success',
                    'amount': amount,
                    'currency': currency,
                    'email': customer_email,
                    'plan': plan_name,
                    'reference': reference,
                    'raw_data': payment_data['data']
                }
            else:
                logger.warning(f"❌ Paystack payment verification failed: {payment_data}")
                return {
                    'status': 'failed',
                    'error': payment_data.get('message', 'Payment verification failed'),
                    'raw_data': payment_data
                }
        else:
            logger.error(f"❌ Paystack API error: {response.status_code} - {response.text}")
            return {
                'status': 'error',
                'error': f'Paystack API error: {response.status_code}',
                'details': response.text
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Network error during Paystack verification: {str(e)}")
        return {
            'status': 'error',
            'error': 'Network error during payment verification',
            'details': str(e)
        }
    except Exception as e:
        logger.error(f"❌ Unexpected error during Paystack verification: {str(e)}")
        return {
            'status': 'error',
            "error": 'Payment verification failed',
            'details': str(e)
        }

async def update_user_credits_paystack(email: str, plan_name: str, amount: float, currency: str):
    """Update user credits based on Paystack payment"""
    try:
        logger.info(f"📝 Updating credits for {email} - {plan_name} ({amount} {currency})")
        
        duration_info = {}
        # Map new plan names to their durations
        if plan_name == 'One-Day Plan':
            duration_info = {'days': 1}
        elif plan_name == 'Three-Day Plan':
            duration_info = {'days': 3}
        elif plan_name == 'One-Week Plan':
            duration_info = {'days': 7}
        
        logger.info(f"✅ Credits updated successfully for {email}")
        return {
            'success': True,
            'email': email,
            'plan': plan_name,
            'amount': amount,
            'currency': currency,
            'duration': duration_info
        }
        
    except Exception as e:
        logger.error(f"❌ Error updating user credits: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
# ===============================================================================
# main.py - Part 4 of 7: Transcription Service Functions (UNCHANGED)
# ===============================================================================

# Transcription service functions
async def transcribe_with_render_whisper(audio_path: str, language_code: str, job_id: str) -> dict:
    """Transcribe audio using the self-hosted Render Whisper backend."""
    if not RENDER_WHISPER_URL:
        logger.error(f"{TYPEMYWORDZ2_NAME} URL not configured, skipping {TYPEMYWORDZ2_NAME} for job {job_id}")
        raise HTTPException(status_code=500, detail=f"{TYPEMYWORDZ2_NAME} URL not configured")

    try:
        logger.info(f"Starting {TYPEMYWORDZ2_NAME} transcription for job {job_id}")
        
        def check_cancellation():
            if job_id and cancellation_flags.get(job_id, False):
                logger.info(f"Job {job_id} was cancelled during {TYPEMYWORDZ2_NAME} processing")
                raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        check_cancellation()

        # Compress audio for Render Whisper
        compressed_path, compression_stats = compress_audio_for_transcription(audio_path, job_id=job_id)
        logger.info(f"Audio compressed for {TYPEMYWORDZ2_NAME}: {compression_stats}")

        # Remove trailing slash from RENDER_WHISPER_URL before appending '/transcribe'
        render_base_url = RENDER_WHISPER_URL.rstrip('/') 

        with open(compressed_path, "rb") as f:
            files = {'file': (os.path.basename(compressed_path), f.read(), 'audio/mpeg')}
            data = {'language_code': language_code}

            # Send request to Render Whisper backend
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{render_base_url}/transcribe", files=files, data=data, timeout=300.0)
                response.raise_for_status()

        result = response.json()
        
        if result.get("status") == "completed" and result.get("transcript"):
            transcription_text = result["transcript"]
            logger.info(f"{TYPEMYWORDZ2_NAME} transcription completed for job {job_id}")
            return {
                "status": "completed",
                "transcription": transcription_text,
                "language": result.get("language", language_code),
                "duration": result.get("duration", 0),
                "word_count": len(transcription_text.split()) if transcription_text else 0,
                "has_speaker_labels": False
            }
        else:
            raise Exception(f"{TYPEMYWORDZ2_NAME} returned an incomplete or failed status: {result}")

    except asyncio.CancelledError:
        logger.info(f"{TYPEMYWORDZ2_NAME} transcription cancelled for job {job_id}")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"{TYPEMYWORDZ2_NAME} HTTP error for job {job_id}: {e.response.status_code} - {e.response.text}")
        return {
            "status": "failed",
            "error": f"{TYPEMYWORDZ2_NAME} HTTP error: {e.response.status_code} - {e.response.text}"
        }
    except httpx.RequestError as e:
        logger.error(f"{TYPEMYWORDZ2_NAME} network error for job {job_id}: {e}")
        return {
            "status": "failed",
            "error": f"{TYPEMYWORDZ2_NAME} network error: {e}"
        }
    except Exception as e:
        logger.error(f"{TYPEMYWORDZ2_NAME} transcription failed for job {job_id}: {str(e)}")
        return {
            "status": "failed",
            "error": f"{TYPEMYWORDZ2_NAME} transcription failed: {str(e)}"
        }
    finally:
        # Clean up compressed file after Render processing
        if 'compressed_path' in locals() and os.path.exists(compressed_path):
            os.unlink(compressed_path)
            logger.info(f"Cleaned up compressed file after {TYPEMYWORDZ2_NAME} processing: {compressed_path}")

async def transcribe_with_assemblyai(audio_path: str, language_code: str, speaker_labels_enabled: bool, model: str, job_id: str) -> dict:
    """Transcribe audio using AssemblyAI API"""
    try:
        logger.info(f"Starting {TYPEMYWORDZ1_NAME} transcription with {model} model for job {job_id}")
        
        def check_cancellation():
            if job_id and cancellation_flags.get(job_id, False):
                logger.info(f"Job {job_id} was cancelled during {TYPEMYWORDZ1_NAME} processing")
                raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        check_cancellation()
        
        # Compress audio for AssemblyAI
        compressed_path, compression_stats = compress_audio_for_transcription(audio_path, job_id=job_id)
        
        check_cancellation()
        
        # Upload to AssemblyAI
        logger.info(f"Uploading audio to {TYPEMYWORDZ1_NAME}...")
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        upload_endpoint = "https://api.assemblyai.com/v2/upload"
        
        with open(compressed_path, "rb") as f:
            upload_response = requests.post(upload_endpoint, headers=headers, data=f)
        
        if upload_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to upload audio to {TYPEMYWORDZ1_NAME}: {upload_response.text}")
        
        upload_result = upload_response.json()
        audio_url = upload_result["upload_url"]
        logger.info(f"Audio uploaded to {TYPEMYWORDZ1_NAME}: {audio_url}")

        check_cancellation()

        # Start transcription
        headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
        transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
        json_data = {
            "audio_url": audio_url,
            "language_code": language_code,
            "punctuate": True,
            "format_text": True,
            "speaker_labels": speaker_labels_enabled,
            "speech_model": model,
            "word_boost": []
        }
        
        transcript_response = requests.post(transcript_endpoint, headers=headers, json=json_data)
        
        if transcript_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to start transcription on {TYPEMYWORDZ1_NAME}: {transcript_response.text}")
        
        transcript_result = transcript_response.json()
        transcript_id = transcript_result["id"]
        logger.info(f"{TYPEMYWORDZ1_NAME} transcription started with ID: {transcript_id}")
        
        # Clean up compressed file
        if os.path.exists(compressed_path):
            os.unlink(compressed_path)
            logger.info(f"Cleaned up compressed file: {compressed_path}")

        # Poll for completion
        while True:
            check_cancellation()
            
            await asyncio.sleep(5)  # Wait 5 seconds between polls
            
            status_response = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers={"authorization": ASSEMBLYAI_API_KEY})
            
            if status_response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Failed to get status from {TYPEMYWORDZ1_NAME}: {status_response.text}")
            
            status_result = status_response.json()
            
            if status_result["status"] == "completed":
                transcription_text = status_result["text"]
                
                # Handle speaker labels with HTML bold formatting
                if speaker_labels_enabled and status_result.get("utterances"):
                    formatted_transcript = ""
                    for utterance in status_result["utterances"]:
                        # Convert Speaker A, B, C to Speaker 1, 2, 3
                        speaker_letter = utterance['speaker']
                        if speaker_letter == 'A':
                            speaker_num = '1'
                        elif speaker_letter == 'B':
                            speaker_num = '2'
                        elif speaker_letter == 'C':
                            speaker_num = '3'
                        elif speaker_letter == 'D':
                            speaker_num = '4'
                        elif speaker_letter == 'E':
                            speaker_num = '5'
                        else:
                            # For any other speakers, convert letter to number
                            speaker_num = str(ord(speaker_letter.upper()) - ord('A') + 1)
                        
                        formatted_transcript += f"<strong>Speaker {speaker_num}:</strong> {utterance['text']}\n"
                    transcription_text = formatted_transcript.strip()

                return {
                    "status": "completed",
                    "transcription": transcription_text,
                    "language": status_result["language_code"],
                    "duration": status_result.get("audio_duration", 0),
                    "word_count": len(transcription_text.split()) if transcription_text else 0,
                    "has_speaker_labels": speaker_labels_enabled and bool(status_result.get("utterances"))
                }
            elif status_result["status"] == "error":
                raise HTTPException(status_code=500, detail=status_result.get("error", f"Transcription failed on {TYPEMYWORDZ1_NAME}"))
            else:
                logger.info(f"{TYPEMYWORDZ1_NAME} status: {status_result['status']}")
                continue
        
    except asyncio.CancelledError:
        logger.info(f"{TYPEMYWORDZ1_NAME} transcription cancelled for job {job_id}")
        raise
    except Exception as e:
        logger.error(f"{TYPEMYWORDZ1_NAME} transcription failed for job {job_id}: {str(e)}")
        return {
            "status": "failed",
            "error": f"{TYPEMYWORDZ1_NAME} transcription failed: {str(e)}"
        }
# ===============================================================================
# main.py - Part 5 of 7: Main Transcription Processing (UPDATED)
# ===============================================================================

# Main transcription processing function
async def process_transcription_job(job_id: str, tmp_path: str, filename: str, language_code: Optional[str], speaker_labels_enabled: bool, user_plan: str, duration_minutes: float):
    """Unified transcription processing with smart model selection and two-tier fallback."""
    logger.info(f"Starting transcription job {job_id}: {filename}, duration: {duration_minutes:.1f}min, plan: {user_plan}")
    job_data = jobs[job_id]
    
    active_background_tasks[job_id] = asyncio.current_task()
    cancellation_flags[job_id] = False

    try:
        def check_cancellation():
            if cancellation_flags.get(job_id, False) or job_data.get("status") == "cancelled":
                logger.info(f"Job {job_id} was cancelled - stopping processing")
                raise asyncio.CancelledError(f"Job {job_id} was cancelled")
            return True

        check_cancellation()

        # UPDATED: Transcription service selection logic
        service_tier_1 = None
        service_tier_2 = None
        
        # Determine AssemblyAI model based on user plan
        def get_assemblyai_model(plan: str) -> str:
            if plan == 'free' or plan == 'One-Day Plan': # Free and One-Day users get nano for TypeMyworDz1 if it were used
                return "nano"  # $0.12/hour
            else:
                return "best"  # $0.27/hour

        assemblyai_model = get_assemblyai_model(user_plan)

        # NEW LOGIC:
        if user_plan == 'free' or user_plan == 'One-Day Plan': # Free and One-Day users only get TypeMyworDz2
            service_tier_1 = "render"
            service_tier_2 = None # No fallback for free/one-day users as per request
            logger.info(f"🎯 User plan '{user_plan}': Primary={TYPEMYWORDZ2_NAME}, No fallback (as per request)")
        else: # All other paid users (Three-Day, One-Week, Pro)
            service_tier_1 = "assemblyai"
            service_tier_2 = "render" # TypeMyworDz2 as fallback for paid users
            logger.info(f"🎯 User plan '{user_plan}': Primary={TYPEMYWORDZ1_NAME} ({assemblyai_model}), Fallback={TYPEMYWORDZ2_NAME}")

        job_data.update({
            "service_tier_1": service_tier_1,
            "service_tier_2": service_tier_2,
            "assemblyai_model": assemblyai_model,
            "duration_minutes": duration_minutes
        })

        transcription_result = None
        services_attempted = []

        # --- ATTEMPT TIER 1 SERVICE ---
        if service_tier_1 == "render":
            if not RENDER_WHISPER_URL:
                logger.error(f"{TYPEMYWORDZ2_NAME} URL not configured, skipping {TYPEMYWORDZ2_NAME} (Tier 1) for job {job_id}")
                job_data["tier_1_error"] = f"{TYPEMYWORDZ2_NAME} URL not configured"
            else:
                try:
                    logger.info(f"🚀 Attempting {TYPEMYWORDZ2_NAME} (Tier 1 Primary) for job {job_id}")
                    transcription_result = await transcribe_with_render_whisper(tmp_path, language_code, job_id)
                    job_data["tier_1_used"] = "render"
                    job_data["tier_1_success"] = True
                except Exception as render_error:
                    logger.error(f"❌ {TYPEMYWORDZ2_NAME} (Tier 1 Primary) failed: {render_error}")
                    job_data["tier_1_error"] = str(render_error)
                    job_data["tier_1_success"] = False
            services_attempted.append(f"{TYPEMYWORDZ2_NAME}_tier1")
        
        elif service_tier_1 == "assemblyai":
            if not ASSEMBLYAI_API_KEY:
                logger.error(f"{TYPEMYWORDZ1_NAME} API Key not configured, skipping {TYPEMYWORDZ1_NAME} (Tier 1) for job {job_id}")
                job_data["tier_1_error"] = f"{TYPEMYWORDZ1_NAME} API Key not configured"
            else:
                try:
                    logger.info(f"🚀 Attempting {TYPEMYWORDZ1_NAME} (Tier 1 Primary) with {assemblyai_model} model for job {job_id}")
                    transcription_result = await transcribe_with_assemblyai(tmp_path, language_code, speaker_labels_enabled, assemblyai_model, job_id)
                    job_data["tier_1_used"] = "assemblyai"
                    job_data["tier_1_success"] = True
                except Exception as assemblyai_error:
                    logger.error(f"❌ {TYPEMYWORDZ1_NAME} (Tier 1 Primary) failed: {assemblyai_error}")
                    job_data["tier_1_error"] = str(assemblyai_error)
                    job_data["tier_1_success"] = False
            services_attempted.append(f"{TYPEMYWORDZ1_NAME}_tier1")

        check_cancellation()

        # --- ATTEMPT TIER 2 SERVICE (FALLBACK 1) if Tier 1 failed AND service_tier_2 is defined ---
        if (not transcription_result or transcription_result.get("status") == "failed") and service_tier_2:
            logger.warning(f"⚠️ Tier 1 service failed, trying Tier 2 fallback ({service_tier_2}) for job {job_id}")
            
            if service_tier_2 == "render":
                if not RENDER_WHISPER_URL:
                    logger.error(f"{TYPEMYWORDZ2_NAME} URL not configured, skipping {TYPEMYWORDZ2_NAME} (Tier 2) for job {job_id}")
                    job_data["tier_2_error"] = f"{TYPEMYWORDZ2_NAME} URL not configured"
                else:
                    try:
                        logger.info(f"🔄 Attempting {TYPEMYWORDZ2_NAME} (Tier 2 Fallback) for job {job_id}")
                        transcription_result = await transcribe_with_render_whisper(tmp_path, language_code, job_id)
                        job_data["tier_2_used"] = "render"
                        job_data["tier_2_success"] = True
                    except Exception as render_error:
                        logger.error(f"❌ {TYPEMYWORDZ2_NAME} (Tier 2 Fallback) failed: {render_error}")
                        job_data["tier_2_error"] = str(render_error)
                        job_data["tier_2_success"] = False
                services_attempted.append(f"{TYPEMYWORDZ2_NAME}_tier2")
            
            elif service_tier_2 == "assemblyai":
                if not ASSEMBLYAI_API_KEY:
                    logger.error(f"{TYPEMYWORDZ1_NAME} API Key not configured, skipping {TYPEMYWORDZ1_NAME} (Tier 2) for job {job_id}")
                    job_data["tier_2_error"] = f"{TYPEMYWORDZ1_NAME} API Key not configured"
                else:
                    try:
                        logger.info(f"🔄 Attempting {TYPEMYWORDZ1_NAME} (Tier 2 Fallback) with {assemblyai_model} model for job {job_id}")
                        transcription_result = await transcribe_with_assemblyai(tmp_path, language_code, speaker_labels_enabled, assemblyai_model, job_id)
                        job_data["tier_2_used"] = "assemblyai"
                        job_data["tier_2_success"] = True
                    except Exception as assemblyai_error:
                        logger.error(f"❌ {TYPEMYWORDZ1_NAME} (Tier 2 Fallback) failed: {assemblyai_error}")
                        job_data["tier_2_error"] = str(assemblyai_error)
                        job_data["tier_2_success"] = False
                services_attempted.append(f"{TYPEMYWORDZ1_NAME}_tier2")

        check_cancellation()

        # Process results
        if transcription_result and transcription_result.get("status") == "completed":
            logger.info(f"✅ Transcription completed successfully for job {job_id}")
            job_data.update({
                "status": "completed",
                "transcription": transcription_result["transcription"],
                "language": transcription_result.get("language", language_code),
                "completed_at": datetime.now().isoformat(),
                "word_count": transcription_result.get("word_count", 0),
                "duration_seconds": transcription_result.get("duration", 0),
                "speaker_labels": transcription_result.get("has_speaker_labels", False),
                "service_used": (job_data.get("tier_1_used") or job_data.get("tier_2_used"))
            })
        else:
            logger.error(f"❌ All active transcription services failed for job {job_id}. Services attempted: {services_attempted}")
            job_data.update({
                "status": "failed",
                "error": f"All active transcription services failed. Services attempted: {', '.join(services_attempted)}",
                "completed_at": datetime.now().isoformat(),
                "services_attempted": services_attempted
            })

    except asyncio.CancelledError:
        logger.info(f"Transcription job {job_id} was cancelled")
        if job_data.get("status") != "cancelled":
            job_data.update({
                "status": "cancelled",
                "cancelled_at": datetime.now().isoformat(),
                "error": "Job was cancelled by user"
            })
        raise
        
    except Exception as e:
        logger.error(f"Transcription job: ERROR during processing for job {job_id}: {str(e)}")
        import traceback
        logger.error(f"Transcription job: Full traceback: {traceback.format_exc()}")
        job_data.update({
            "status": "failed",
            "error": f"Internal server error during transcription: {str(e)}",
            "completed_at": datetime.now().isoformat()
        })
    finally:
        if os.path.exists(tmp_path):
            logger.info(f"Transcription job: Cleaning up temporary file: {tmp_path}")
            os.unlink(tmp_path)
        
        if job_id in active_background_tasks:
            del active_background_tasks[job_id]
        if job_id in cancellation_flags:
            del cancellation_flags[job_id]
            
        logger.info(f"Transcription job completed for job ID: {job_id}")
# ===============================================================================
# main.py - Part 6 of 7: App Setup and AI Endpoints (UPDATED)
# ===============================================================================

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application lifespan startup")
    health_task = asyncio.create_task(health_monitor())
    logger.info("Health monitor task created")
    yield
    logger.info("Application lifespan shutdown")
    health_task.cancel()
    for job_id, task in active_background_tasks.items():
        if not task.done():
            logger.info(f"Cancelling background task for job {job_id}")
            cancellation_flags[job_id] = True
            task.cancel()
    jobs.clear()
    active_background_tasks.clear()
    cancellation_flags.clear()
    logger.info("All background tasks cancelled and cleanup complete")

logger.info("Creating FastAPI app...")
app = FastAPI(title=f"Enhanced Transcription Service with {TYPEMYWORDZ1_NAME}, {TYPEMYWORDZ2_NAME} & {TYPEMYWORDZ_AI_NAME}", lifespan=lifespan)
logger.info("FastAPI app created successfully")

logger.info("Setting up CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured successfully")

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {
        "message": f"Enhanced Transcription Service with {TYPEMYWORDZ1_NAME}, {TYPEMYWORDZ2_NAME} & {TYPEMYWORDZ_AI_NAME} is running!",
        "features": [
            f"{TYPEMYWORDZ2_NAME} integration",
            f"{TYPEMYWORDZ1_NAME} integration with smart model selection",
            "Two-tier automatic fallback between services",
            "Paystack payment integration",
            f"Speaker diarization for {TYPEMYWORDZ1_NAME}",
            "Language selection for transcription",
            f"{TYPEMYWORDZ_AI_NAME} for user-driven summarization, Q&A, and bullet points",
            f"{TYPEMYWORDZ_AI_NAME} for admin-driven transcript formatting"
        ],
        "logic": {
            "free_user_transcription": f"Only {TYPEMYWORDZ2_NAME}",
            "paid_user_transcription": f"Primary={TYPEMYWORDZ1_NAME} → {TYPEMYWORDZ2_NAME} Fallback",
            "free_users_assemblyai": f"{TYPEMYWORDZ1_NAME} nano model",
            "paid_users_assemblyai": f"{TYPEMYWORDZ1_NAME} best model",
            "ai_features_access": "Only for Three-Day and Pro plans" # UPDATED
        },
        "stats": {
            "active_jobs": len(jobs),
            "background_tasks": len(active_background_tasks),
            "cancellation_flags": len(cancellation_flags)
        }
    }

# ADDED: Quick Test Method for Claude Models
@app.get("/test-claude-models")
async def test_claude_models():
    """Test which Claude models work with your API key"""
    if not claude_client:
        return {"error": "Claude client not initialized - check your ANTHROPIC_API_KEY"}
    
    models_to_test = [
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022"
    ]
    
    results = {}
    
    for model in models_to_test:
        try:
            logger.info(f"Testing Claude model: {model}")
            message = claude_client.messages.create(
                model=model,
                max_tokens=10,
                timeout=10.0,
                messages=[{"role": "user", "content": "Hello"}]
            )
            results[model] = "✅ WORKS"
            logger.info(f"✅ Model {model} works!")
        except anthropic.APIError as e:
            error_msg = str(e)
            if hasattr(e, 'body'):
                try:
                    error_data = e.body if isinstance(e.body, dict) else {"error": str(e.body)}
                    error_msg = str(error_data)
                except:
                    pass
            results[model] = f"❌ API ERROR: {error_msg}"
            logger.error(f"❌ Model {model} failed: {error_msg}")
        except Exception as e:
            results[model] = f"❌ ERROR: {str(e)}"
            logger.error(f"❌ Model {model} failed: {str(e)}")
    
    return {
        "test_results": results,
        "recommendation": "Use the first model marked with ✅ WORKS in your UserAIRequest and AdminAIFormatRequest classes",
        "current_default_models": {
            "UserAIRequest": "claude-3-haiku-20240307",
            "AdminAIFormatRequest": "claude-3-5-haiku-20241022"
        }
    }

@app.post("/api/initialize-paystack-payment")
async def initialize_paystack_payment(request: PaystackInitializationRequest):
    logger.info(f"Initializing Paystack payment for {request.email} in {request.country_code}: Base USD {request.amount}")
    
    if not PAYSTACK_SECRET_KEY:
        logger.error("❌ PAYSTACK_SECRET_KEY is not set in environment variables.")
        raise HTTPException(status_code=500, detail="Paystack configuration missing")
    
    try:
        local_amount, local_currency = get_local_amount_and_currency(request.amount, request.country_code)
        payment_channels = get_payment_channels(request.country_code)

        amount_kobo = int(local_amount * 100)
        
        headers = {
            'Authorization': f'Bearer {PAYSTACK_SECRET_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'email': request.email,
            'amount': amount_kobo,
            'currency': local_currency,
            'callback_url': request.callback_url,
            'channels': payment_channels,
            'metadata': {
                'plan': request.plan_name,
                'base_usd_amount': request.amount,
                'custom_fields': [
                    {
                        'display_name': "Plan Type",
                        'variable_name': "plan_type",
                        'value': request.plan_name
                    },
                    {
                        'display_name': "Country",
                        'variable_name': "country",
                        'value': request.country_code
                    }
                ]
            }
        }
        
        logger.info(f"DEBUG: Paystack payload for {request.country_code}: Amount={local_amount} {local_currency}, Channels={payment_channels}")

        response = requests.post(
            'https://api.paystack.co/transaction/initialize',
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Paystack payment initialized: {result['data']['reference']}")
            return {
                'status': True,
                'authorization_url': result['data']['authorization_url'],
                'reference': result['data']['reference'],
                'local_amount': local_amount,
                'local_currency': local_currency
            }
        else:
            logger.error(f"❌ Paystack API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Paystack API error: {response.text}")
            
    except Exception as e:
        import traceback
        logger.error(f"❌ Error initializing Paystack payment: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Payment initialization failed: {str(e)}")

@app.post("/api/verify-payment")
async def verify_payment(request: PaystackVerificationRequest):
    logger.info(f"Payment verification request for reference: {request.reference}")
    
    try:
        verification_result = await verify_paystack_payment(request.reference)
        
        if verification_result['status'] == 'success':
            email = verification_result['email']
            plan_name = verification_result['plan']
            amount = verification_result['amount']
            currency = verification_result['currency']
            
            credit_result = await update_user_credits_paystack(email, plan_name, amount, currency)
            
            if credit_result['success']:
                logger.info(f"✅ Payment verified and credits updated for {email}")
                return {
                    "status": "success",
                    "message": "Payment verified successfully",
                    "data": {
                        "amount": amount,
                        "currency": currency,
                        "email": email,
                        "plan": plan_name,
                        "reference": request.reference,
                        "credits_updated": True
                    }
                }
            else:
                logger.warning(f"⚠️ Payment verified but credit update failed for {email}")
                return {
                    "status": "partial_success",
                    "message": "Payment verified but credit update failed",
                    "data": {
                        "amount": amount,
                        "currency": currency,
                        "email": email,
                        "plan": plan_name,
                        "reference": request.reference,
                        "credits_updated": False,
                        "credit_error": credit_result.get('error')
                    }
                }
        else:
            logger.warning(f"❌ Payment verification failed for reference: {request.reference}")
            raise HTTPException(
                status_code=400, 
                detail=verification_result.get('error', 'Payment verification failed'),
                headers={"X-Error-Details": verification_result.get('details', '')}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during payment verification: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Payment verification failed: {str(e)}"
        )

@app.post("/api/paystack-webhook")
async def paystack_webhook(request: Request):
    try:
        body = await request.body()
        signature = request.headers.get('x-paystack-signature')
        
        logger.info(f"Received Paystack webhook with signature: {bool(signature)}")
        
        if not body:
            logger.warning("Empty webhook body received")
            raise HTTPException(status_code=400, detail="Empty webhook body")
        
        try:
            webhook_data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in webhook: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        
        event_type = webhook_data.get('event')
        logger.info(f"Processing Paystack webhook event: {event_type}")
        
        if event_type == 'charge.success':
            data = webhook_data.get('data', {})
            customer_email = data.get('customer', {}).get('email')
            amount = data.get('amount', 0) / 100
            currency = data.get('currency')
            reference = data.get('reference')
            plan_name = data.get('metadata', {}).get('plan', 'Unknown')
            
            logger.info(f"🔔 Webhook: Payment successful - {customer_email} paid {amount} {currency} for {plan_name}")
            
            if customer_email:
                credit_result = await update_user_credits_paystack(customer_email, plan_name, amount, currency)
                if credit_result['success']:
                    logger.info(f"✅ Webhook: Credits updated automatically for {customer_email}")
                else:
                    logger.warning(f"⚠️ Webhook: Failed to update credits for {customer_email}")
            
        elif event_type == 'charge.failed':
            data = webhook_data.get('data', {})
            customer_email = data.get('customer', {}).get('email')
            reference = data.get('reference')
            
            logger.warning(f"🔔 Webhook: Payment failed for {customer_email}, reference: {reference}")
            
        else:
            logger.info(f"🔔 Webhook: Unhandled event type: {event_type}")
        
        return {"status": "received", "event": event_type}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing Paystack webhook: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Webhook processing failed: {str(e)}")

@app.get("/api/paystack-status")
async def paystack_status():
    return {
        "paystack_configured": bool(PAYSTACK_SECRET_KEY),
        "public_key_configured": bool(PAYSTACK_PUBLIC_KEY),
        "webhook_secret_configured": bool(PAYSTACK_WEBHOOK_SECRET),
        "assemblyai_configured": bool(ASSEMBLYAI_API_KEY),
        "render_whisper_configured": bool(RENDER_WHISPER_URL),
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "endpoints": {
            "initialize_payment": "/api/initialize-paystack-payment",
            "verify_payment": "/api/verify-payment",
            "webhook": "/api/paystack-webhook",
            "status": "/api/paystack-status",
            "transcribe": "/transcribe",
            "ai_user_query": "/ai/user-query",
            "ai_admin_format": "/ai/admin-format",
            "test_claude_models": "/test-claude-models"
        },
        "supported_currencies": ["NGN", "USD", "GHS", "ZAR", "KES"],
        "supported_plans": [
            "One-Day Plan",
            "Three-Day Plan",
            "One-Week Plan",
            "Pro"
        ],
        "conversion_rates_usd_to_local": USD_TO_LOCAL_RATES
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form("en"),
    speaker_labels_enabled: bool = Form(False),
    user_plan: str = Form("free"), # Get user_plan from frontend
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    logger.info(f"Main transcription endpoint called with file: {file.filename}, language: {language_code}, speaker_labels: {speaker_labels_enabled}, user_plan: {user_plan}")
    
    if not file.content_type.startswith(('audio/', 'video/')):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Please upload an audio or video file")
    
    job_id = str(uuid.uuid4())
    logger.info(f"Created transcription job ID: {job_id}")
    
    try:
        logger.info(f"Saving uploaded file {file.filename} temporarily...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        file_size_mb = len(content) / (1024 * 1024)
        
        # Analyze audio to get duration
        audio_characteristics = await analyze_audio_characteristics(tmp_path)
        duration_seconds = audio_characteristics.get("duration_seconds", 0)
        duration_minutes = duration_seconds / 60.0
        
        logger.info(f"Audio analysis: {duration_minutes:.1f} minutes, {file_size_mb:.2f} MB")
        
        jobs[job_id] = {
            "status": "processing",
            "filename": file.filename,
            "created_at": datetime.now().isoformat(),
            "file_size_mb": round(file_size_mb, 2),
            "content_type": file.content_type,
            "requested_language": language_code,
            "speaker_labels_enabled": speaker_labels_enabled,
            "user_plan": user_plan,
            "duration_minutes": duration_minutes,
            "duration_seconds": duration_seconds
        }
        
        cancellation_flags[job_id] = False
        logger.info(f"Job {job_id} initialized with status 'processing'")
        
    except Exception as e:
        logger.error(f"ERROR processing file for job {job_id}: {str(e)}")
        if job_id in jobs:
            del jobs[job_id]
        if job_id in cancellation_flags:
            del cancellation_flags[job_id]
        raise HTTPException(status_code=500, detail="Failed to process audio file")

    background_tasks.add_task(
        process_transcription_job, 
        job_id, 
        tmp_path, 
        file.filename, 
        language_code, 
        speaker_labels_enabled, 
        user_plan, 
        duration_minutes
    )
    
    logger.info(f"Returning immediate response for job ID: {job_id}")
    return {
        "job_id": job_id, 
        "status": jobs[job_id]["status"],
        "filename": file.filename,
        "file_size_mb": jobs[job_id]["file_size_mb"],
        "duration_minutes": duration_minutes,
        "created_at": jobs[job_id]["created_at"],
        "logic_used": f"UserPlan:{user_plan}" # UPDATED: Logic description
    }

@app.post("/generate-formatted-word")
async def generate_formatted_word(request: FormattedWordDownloadRequest):
    logger.info(f"Generating formatted Word document for {request.filename}")
    try:
        document = Document()
        # Split the HTML by lines to process each utterance
        lines = request.transcription_html.split('\n')
        
        # Regex to find speaker tags: <strong>Speaker X:</strong>
        speaker_tag_pattern = re.compile(r'<strong>(Speaker \d+:)</strong>(.*)')
        
        for line in lines:
            if line.strip(): # Process non-empty lines
                p = document.add_paragraph()
                
                match = speaker_tag_pattern.match(line)
                if match:
                    speaker_label_text = match.group(1)
                    remaining_text = match.group(2).strip()

                    # Add speaker label as bold
                    run = p.add_run(speaker_label_text)
                    run.bold = True
                    
                    # Add the rest of the text as normal
                    if remaining_text:
                        p.add_run(" " + remaining_text)
                else:
                    # No speaker tag found, add as plain text (stripping any other HTML tags)
                    clean_line = re.sub(r'<[^>]*>', '', line).strip() # Strip all HTML tags
                    if clean_line:
                        p.add_run(clean_line)

        # Save document to a BytesIO object
        file_stream = BytesIO()
        document.save(file_stream)
        file_stream.seek(0) # Rewind to the beginning of the stream

        return StreamingResponse(
            file_stream,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename={request.filename}"}
        )

    except Exception as e:
        logger.error(f"Error generating formatted Word document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate formatted Word document: {str(e)}")

# UPDATED: AI Endpoint for User Queries with plan check
@app.post("/ai/user-query")
async def ai_user_query(request: UserAIRequest, user_plan: str = Form("free")): # Get user_plan from frontend
    logger.info(f"AI user query endpoint called. Model: {request.model}, Prompt: '{request.user_prompt}', User Plan: {user_plan}")

    # NEW: AI Access Control
    if not is_paid_ai_user(user_plan):
        raise HTTPException(status_code=403, detail="AI Assistant features are only available for paid AI users (Three-Day, Pro plans). Please upgrade your plan.")

    if not claude_client:
        raise HTTPException(status_code=503, detail=f"{TYPEMYWORDZ_AI_NAME} service is not initialized (API key missing or invalid).")

    try:
        # Validate input lengths
        if len(request.transcript) > 100000:  # ~100k chars limit
            raise HTTPException(status_code=400, detail="Transcript is too long. Please use a shorter transcript.")
        
        if len(request.user_prompt) > 1000:  # 1k chars limit for prompts
            raise HTTPException(status_code=400, detail="User prompt is too long. Please use a shorter prompt.")

        # Construct the full prompt for Claude
        full_prompt = f"{request.user_prompt}\n\nHere is the transcript:\n{request.transcript}"

        message = claude_client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            timeout=30.0,  # Add timeout
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        ai_response = message.content[0].text
        logger.info(f"Successfully processed AI user query with {request.model}.")
        return {"ai_response": ai_response}

    except anthropic.APIError as e:
        # Improved error handling for Anthropic API errors
        error_message = "AI service error"
        error_details = str(e)
        
        if hasattr(e, 'body'):
            try:
                error_data = e.body if isinstance(e.body, dict) else {"error": str(e.body)}
                error_details = error_data
                logger.error(f"Anthropic API Error for user query: {error_data}")
            except:
                logger.error(f"Anthropic API Error for user query: {str(e)}")
        else:
            logger.error(f"Anthropic API Error for user query: {str(e)}")
            
        raise HTTPException(status_code=500, detail=f"{error_message}: {error_details}")
    
    except anthropic.APITimeoutError as e:
        logger.error(f"Anthropic API Timeout for user query: {e}")
        raise HTTPException(status_code=504, detail="AI service timeout. Please try again.")
    
    except Exception as e:
        logger.error(f"Unexpected error processing AI user query: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# UPDATED: AI Endpoint for Admin Formatting with plan check
@app.post("/ai/admin-format")
async def ai_admin_format(request: AdminAIFormatRequest, user_plan: str = Form("free")): # Get user_plan from frontend
    logger.info(f"AI admin format endpoint called. Model: {request.model}, Instructions: '{request.formatting_instructions}', User Plan: {user_plan}")

    # NEW: AI Access Control
    if not is_paid_ai_user(user_plan):
        raise HTTPException(status_code=403, detail="AI Admin formatting features are only available for paid AI users (Three-Day, Pro plans). Please upgrade your plan.")

    if not claude_client:
        raise HTTPException(status_code=503, detail=f"{TYPEMYWORDZ_AI_NAME} service is not initialized (API key missing or invalid).")

    try:
        # Validate input lengths
        if len(request.transcript) > 200000:  # ~200k chars limit for admin (higher than user)
            raise HTTPException(status_code=400, detail="Transcript is too long. Please use a shorter transcript.")
        
        if len(request.formatting_instructions) > 2000:  # 2k chars limit for admin instructions
            raise HTTPException(status_code=400, detail="Formatting instructions are too long. Please use shorter instructions.")

        # Construct the full prompt for Claude, emphasizing the formatting instructions
        full_prompt = f"Please apply the following formatting and polishing instructions to the provided transcript:\n\nInstructions: {request.formatting_instructions}\n\nTranscript to format:\n{request.transcript}"

        message = claude_client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            timeout=60.0,  # Longer timeout for admin tasks
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        ai_response = message.content[0].text
        logger.info(f"Successfully processed AI admin format request with {request.model}.")
        return {"formatted_transcript": ai_response}

    except anthropic.APIError as e:
        # Improved error handling for Anthropic API errors
        error_message = "AI service error for admin formatting"
        error_details = str(e)
        
        if hasattr(e, 'body'):
            try:
                error_data = e.body if isinstance(e.body, dict) else {"error": str(e.body)}
                error_details = error_data
                logger.error(f"Anthropic API Error for admin format: {error_data}")
            except:
                logger.error(f"Anthropic API Error for admin format: {str(e)}")
        else:
            logger.error(f"Anthropic API Error for admin format: {str(e)}")
            
        raise HTTPException(status_code=500, detail=f"{error_message}: {error_details}")
    
    except anthropic.APITimeoutError as e:
        logger.error(f"Anthropic API Timeout for admin format: {e}")
        raise HTTPException(status_code=504, detail="AI service timeout. Please try again with a shorter transcript.")
    
    except Exception as e:
        logger.error(f"Unexpected error processing AI admin format request: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during admin formatting: {str(e)}")
# ===============================================================================
# main.py - Part 7 of 7: Final Endpoints and Application Startup (UPDATED)
# ===============================================================================

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    logger.info(f"Status check for job ID: {job_id}")
    if job_id not in jobs:
        logger.warning(f"Job ID {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs[job_id]
    
    if job_data["status"] == "cancelled" or cancellation_flags.get(job_id, False):
        logger.info(f"Job {job_id} was cancelled, returning cancelled status")
        job_data["status"] = "cancelled"
    
    return job_data

@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    logger.info(f"Cancel request received for job ID: {job_id}")
    
    if job_id not in jobs:
        logger.warning(f"Cancel request: Job ID {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs[job_id]
    
    try:
        cancellation_flags[job_id] = True
        logger.info(f"Cancellation flag set for job {job_id}")
        
        if job_id in active_background_tasks:
            task = active_background_tasks[job_id]
            if not task.done():
                logger.info(f"Cancelling active background task for job {job_id}")
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.info(f"Background task for job {job_id} cancelled (timeout/cancelled)")
            else:
                logger.info(f"Background task for job {job_id} was already completed")
        
        job_data.update({
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat(),
            "error": "Job was cancelled by user"
        })
        
        logger.info(f"Job {job_id} successfully cancelled")
        return {
            "message": "Job cancelled successfully", 
            "job_id": job_id,
            "cancelled_at": job_data["cancelled_at"],
            "previous_status": job_data.get("previous_status", "processing")
        }
        
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}")
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            input_path = tmp.name
        
        output_path = compress_audio_for_download(input_path, quality=quality)
        
        with open(output_path, 'rb') as f:
            compressed_content = f.read()
        
        os.unlink(input_path)
        os.unlink(output_path)
        
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
    logger.info("Cleanup endpoint called")
    
    current_time = datetime.now()
    jobs_to_remove = []
    tasks_to_cancel = []
    flags_to_remove = []
    
    for job_id, job_data in jobs.items():
        created_at = datetime.fromisoformat(job_data["created_at"])
        age_hours = (current_time - created_at).total_seconds() / 3600
        
        if age_hours > 1 and job_data["status"] in ["completed", "failed", "cancelled"]:
            jobs_to_remove.append(job_id)
            
            if job_id in active_background_tasks:
                task = active_background_tasks[job_id]
                if not task.done():
                    tasks_to_cancel.append((job_id, task))
                    
            if job_id in cancellation_flags:
                flags_to_remove.append(job_id)
    
    for job_id, task in tasks_to_cancel:
        try:
            task.cancel()
            logger.info(f"Cancelled old background task for job: {job_id}")
        except Exception as e:
            logger.error(f"Error cancelling old task {job_id}: {e}")
    
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
    logger.info("Jobs list endpoint called")
    
    job_summary = {}
    for job_id, job_data in jobs.items():
        job_summary[job_id] = {
            "status": job_data["status"],
            "filename": job_data.get("filename", "unknown"),
            "created_at": job_data["created_at"],
            "file_size_mb": job_data.get("file_size_mb", 0),
            "duration_minutes": job_data.get("duration_minutes", 0),
            "user_plan": job_data.get("user_plan", "unknown"),
            "primary_service": job_data.get("primary_service"),
            "service_used": job_data.get("service_used"),
            "has_background_task": job_id in active_background_tasks,
            "is_cancellation_flagged": cancellation_flags.get(job_id, False),
            "word_count": job_data.get("word_count"),
            "duration_seconds": job_data.get("duration_seconds"),
            "requested_language": job_data.get("requested_language", "en"),
            "speaker_labels_enabled": job_data.get("speaker_labels_enabled", False)
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
            },
            "integrations": {
                "assemblyai_configured": bool(ASSEMBLYAI_API_KEY),
                "render_whisper_configured": bool(RENDER_WHISPER_URL),
                "anthropic_configured": bool(ANTHROPIC_API_KEY),
                "paystack_configured": bool(PAYSTACK_SECRET_KEY)
            },
            "transcription_logic": {
                "free_user_transcription": f"Only {TYPEMYWORDZ2_NAME}",
                "paid_user_transcription": f"Primary={TYPEMYWORDZ1_NAME} → {TYPEMYWORDZ2_NAME} Fallback",
                "free_users_assemblyai": f"{TYPEMYWORDZ1_NAME} nano model ($0.12/hour)",
                "paid_users_assemblyai": f"{TYPEMYWORDZ1_NAME} best model ($0.27/hour)",
                "ai_features_access": "Only for Three-Day and Pro plans", # UPDATED
                "render_whisper": f"{TYPEMYWORDZ2_NAME} (self-hosted Whisper)",
                "ai_features": f"{TYPEMYWORDZ_AI_NAME} (Anthropic Claude 3 Haiku / 3.5 Haiku)"
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

logger.info("Performing final system validation...")
logger.info(f"{TYPEMYWORDZ1_NAME} API Key configured: {bool(ASSEMBLYAI_API_KEY)}")
logger.info(f"{TYPEMYWORDZ2_NAME} URL configured: {bool(RENDER_WHISPER_URL)}")
logger.info(f"{TYPEMYWORDZ_AI_NAME} API Key configured: {bool(ANTHROPIC_API_KEY)}")
logger.info(f"Paystack Secret Key configured: {bool(PAYSTACK_SECRET_KEY)}")

logger.info("Available API endpoints:")
logger.info("  POST /transcribe - Main transcription endpoint with smart service selection")
logger.info("  GET /test-claude-models - Test which Claude models work with your API key")
logger.info("  POST /ai/user-query - Process user-driven AI queries (summarize, Q&A, bullet points)")
logger.info("  POST /ai/admin-format - Process admin-driven AI formatting requests")
logger.info("  POST /api/initialize-paystack-payment - Initialize Paystack payment")
logger.info("  POST /api/verify-payment - Verify Paystack payment")
logger.info("  POST /api/paystack-webhook - Handle Paystack webhooks")
logger.info("  GET /api/paystack-status - Get integration status")
logger.info("  GET /status/{job_id} - Check job status")
logger.info("  POST /cancel/{job_id} - Cancel transcription job")
logger.info("  POST /compress-download - Compress audio for download")
logger.info("  POST /generate-formatted-word - Generate formatted Word document with speaker labels")
logger.info("  GET /jobs - List all jobs")
logger.info("  GET /health - System health check")
logger.info("  DELETE /cleanup - Clean up old jobs")
logger.info("  GET / - Root endpoint with service info")

if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly...")
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting enhanced transcription service on {host}:{port}")
    logger.info("🚀 NEW ENHANCED FEATURES:")
    logger.info(f"  ✅ {TYPEMYWORDZ2_NAME} re-integrated")
    logger.info(f"  ✅ Smart service selection with updated logic")
    logger.info(f"  ✅ Two-tier automatic fallback for paid users")
    logger.info(f"  ✅ Speaker diarization for {TYPEMYWORDZ1_NAME}")
    logger.info(f"  ✅ Dynamic {TYPEMYWORDZ1_NAME} model selection (nano for free, best for paid)")
    logger.info("  ✅ Unified transcription processing pipeline")
    logger.info("  ✅ Enhanced error handling and service resilience")
    logger.info("  ✅ Comprehensive job tracking and cancellation")
    logger.info("  ✅ Paystack payment integration")
    logger.info("  ✅ Multi-language support")
    logger.info("  ✅ Formatted Word document generation")
    logger.info(f"  ✅ User-driven AI features (summarization, Q&A, bullet points) via {TYPEMYWORDZ_AI_NAME}")
    logger.info(f"  ✅ Admin-driven AI formatting via {TYPEMYWORDZ_AI_NAME}")
    logger.info(f"  ✅ AI Assistant features restricted to paid users (Three-Day, Pro plans)") # UPDATED
    
    logger.info("🔧 TRANSCRIPTION LOGIC:")
    logger.info(f"  - Free/One-Day users: Only {TYPEMYWORDZ2_NAME}")
    logger.info(f"  - Paid users (3-Day, 1-Week, Pro): {TYPEMYWORDZ1_NAME} primary → {TYPEMYWORDZ2_NAME} fallback")
    logger.info(f"  - Free users: {TYPEMYWORDZ1_NAME} nano model ($0.12/hour)")
    logger.info(f"  - Paid users: {TYPEMYWORDZ1_NAME} best model ($0.27/hour)")
    logger.info(f"  - {TYPEMYWORDZ2_NAME}: self-hosted Whisper (variable cost)")
    logger.info(f"  - {TYPEMYWORDZ1_NAME} supports speaker labels with HTML formatting")
    logger.info(f"  - {TYPEMYWORDZ2_NAME} typically does NOT support speaker labels")
    logger.info(f"  - {TYPEMYWORDZ_AI_NAME} (Anthropic Claude 3 Haiku / 3.5 Haiku) for text processing")
    
    try:
        uvicorn.run(
            app, 
            host=host, 
            port=port,
            log_level="info",
            access_log=True,
            reload=False,
            workers=1
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
else:
    logger.info("Application loaded as module")
    logger.info(f"Ready to handle requests with {TYPEMYWORDZ1_NAME} + {TYPEMYWORDZ2_NAME} + {TYPEMYWORDZ_AI_NAME} integration")

# ===============================================================================
# END COMPLETE UPDATED main.py
# ===============================================================================
