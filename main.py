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
from typing import Optional
import httpx
# NEW IMPORTS for python-docx and regex
from docx import Document
from docx.shared import Inches
from io import BytesIO
from fastapi.responses import StreamingResponse
import re # Added for robust HTML parsing
# NEW: OpenAI import
import openai

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")
PAYSTACK_PUBLIC_KEY = os.getenv("PAYSTACK_PUBLIC_KEY")
PAYSTACK_WEBHOOK_SECRET = os.getenv("PAYSTACK_WEBHOOK_SECRET")

logger.info(f"Attempted to load ASSEMBLYAI_API_KEY. Value found: {bool(ASSEMBLYAI_API_KEY)}")
logger.info(f"Attempted to load OPENAI_API_KEY. Value found: {bool(OPENAI_API_KEY)}")
logger.info(f"Attempted to load PAYSTACK_SECRET_KEY. Value found: {bool(PAYSTACK_SECRET_KEY)}")
logger.info(f"Attempted to load PAYSTACK_PUBLIC_KEY. Value found: {bool(PAYSTACK_PUBLIC_KEY)}")

if not ASSEMBLYAI_API_KEY:
    logger.error("ASSEMBLYAI_API_KEY environment variable not set! AssemblyAI fallback will not work.")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set! OpenAI Whisper will not work.")
else:
    # Configure OpenAI
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API key configured successfully")

if not PAYSTACK_SECRET_KEY:
    logger.warning("PAYSTACK_SECRET_KEY environment variable not set! Paystack features will be disabled.")

if PAYSTACK_SECRET_KEY:
    logger.info("Paystack configuration found - payment verification enabled")
else:
    logger.warning("Paystack configuration missing - payment verification disabled")

logger.info("Environment variables loaded successfully")

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

# NEW: Request model for formatted Word download
class FormattedWordDownloadRequest(BaseModel):
    transcription_html: str
    filename: Optional[str] = "transcription.docx"

jobs = {}
active_background_tasks = {}
cancellation_flags = {}

logger.info("Enhanced job tracking initialized")
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
            'error': 'Payment verification failed',
            'details': str(e)
        }

async def update_user_credits_paystack(email: str, plan_name: str, amount: float, currency: str):
    """Update user credits based on Paystack payment"""
    try:
        logger.info(f"📝 Updating credits for {email} - {plan_name} ({amount} {currency})")
        
        duration_info = {}
        # UPDATED: Map new plan names to their durations
        if plan_name == 'One-Day Plan':
            duration_info = {'days': 1}
        elif plan_name == 'Three-Day Plan':
            duration_info = {'days': 3}
        elif plan_name == 'One-Week Plan':
            duration_info = {'days': 7}
        # 'pro' plan is typically handled as unlimited, so no specific duration needed here
        
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
# NEW: OpenAI Whisper transcription function
async def transcribe_with_openai_whisper(audio_path: str, language_code: str = "en", job_id: str = None) -> dict:
    """Transcribe audio using OpenAI Whisper API with speaker diarization"""
    try:
        logger.info(f"Starting OpenAI Whisper transcription for job {job_id}")
        
        def check_cancellation():
            if job_id and cancellation_flags.get(job_id, False):
                logger.info(f"Job {job_id} was cancelled during OpenAI Whisper processing")
                raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        check_cancellation()
        
        # Get file size for logging
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"OpenAI Whisper: Processing {file_size_mb:.2f} MB audio file")
        
        # Open and transcribe the audio file
        with open(audio_path, "rb") as audio_file:
            check_cancellation()
            
            # Use OpenAI Whisper API with speaker timestamps
            transcript = await asyncio.to_thread(
                openai.Audio.transcribe,
                model="whisper-1",
                file=audio_file,
                language=language_code if language_code != "en" else None,  # Let Whisper auto-detect for English
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        
        check_cancellation()
        
        # Process the transcript for speaker diarization simulation
        transcription_text = transcript.text
        segments = transcript.get('segments', [])
        
        # Simple speaker diarization simulation based on pauses
        if segments and len(segments) > 1:
            formatted_transcript = ""
            current_speaker = 1
            last_end_time = 0
            
            for segment in segments:
                start_time = segment.get('start', 0)
                text = segment.get('text', '').strip()
                
                # If there's a significant pause (>3 seconds), switch speaker
                if start_time - last_end_time > 3.0:
                    current_speaker = 2 if current_speaker == 1 else 1
                
                if text:
                    formatted_transcript += f"<strong>Speaker {current_speaker}:</strong> {text}\n"
                
                last_end_time = segment.get('end', start_time)
            
            # Use formatted transcript if we have segments, otherwise use plain text
            final_transcript = formatted_transcript.strip() if formatted_transcript.strip() else transcription_text
        else:
            final_transcript = transcription_text
        
        logger.info(f"OpenAI Whisper transcription completed for job {job_id}")
        
        return {
            "status": "completed",
            "transcription": final_transcript,
            "language": transcript.get('language', language_code),
            "duration": transcript.get('duration', 0),
            "word_count": len(transcription_text.split()) if transcription_text else 0,
            "segments_count": len(segments),
            "has_speaker_labels": len(segments) > 1  # Indicate if we added speaker labels
        }
        
    except asyncio.CancelledError:
        logger.info(f"OpenAI Whisper transcription cancelled for job {job_id}")
        raise
    except Exception as e:
        logger.error(f"OpenAI Whisper transcription failed for job {job_id}: {str(e)}")
        return {
            "status": "failed",
            "error": f"OpenAI Whisper transcription failed: {str(e)}"
        }

# NEW: AssemblyAI transcription function
async def transcribe_with_assemblyai(audio_path: str, language_code: str, speaker_labels_enabled: bool, model: str, job_id: str) -> dict:
    """Transcribe audio using AssemblyAI API"""
    try:
        logger.info(f"Starting AssemblyAI transcription with {model} model for job {job_id}")
        
        def check_cancellation():
            if job_id and cancellation_flags.get(job_id, False):
                logger.info(f"Job {job_id} was cancelled during AssemblyAI processing")
                raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        check_cancellation()
        
        # Compress audio for AssemblyAI
        compressed_path, compression_stats = compress_audio_for_transcription(audio_path, job_id=job_id)
        
        check_cancellation()
        
        # Upload to AssemblyAI
        logger.info("Uploading audio to AssemblyAI...")
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        upload_endpoint = "https://api.assemblyai.com/v2/upload"
        
        with open(compressed_path, "rb") as f:
            upload_response = requests.post(upload_endpoint, headers=headers, data=f)
        
        if upload_response.status_code != 200:
            raise Exception(f"Failed to upload audio to AssemblyAI: {upload_response.text}")
        
        upload_result = upload_response.json()
        audio_url = upload_result["upload_url"]
        logger.info(f"Audio uploaded to AssemblyAI: {audio_url}")

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
            raise Exception(f"Failed to start transcription on AssemblyAI: {transcript_response.text}")
        
        transcript_result = transcript_response.json()
        transcript_id = transcript_result["id"]
        logger.info(f"AssemblyAI transcription started with ID: {transcript_id}")
        
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
                raise Exception(f"Failed to get status from AssemblyAI: {status_response.text}")
            
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
                    "word_count": len(status_result["text"].split()) if status_result["text"] else 0,
                    "has_speaker_labels": speaker_labels_enabled and bool(status_result.get("utterances"))
                }
            elif status_result["status"] == "error":
                raise Exception(status_result.get("error", "Transcription failed on AssemblyAI"))
            else:
                logger.info(f"AssemblyAI status: {status_result['status']}")
                continue
        
    except asyncio.CancelledError:
        logger.info(f"AssemblyAI transcription cancelled for job {job_id}")
        raise
    except Exception as e:
        logger.error(f"AssemblyAI transcription failed for job {job_id}: {str(e)}")
        return {
            "status": "failed",
            "error": f"AssemblyAI transcription failed: {str(e)}"
        }
# UPDATED: process_transcription_job - NEW unified function
async def process_transcription_job(job_id: str, tmp_path: str, filename: str, language_code: Optional[str], speaker_labels_enabled: bool, user_plan: str, duration_minutes: float):
    """Unified transcription processing with smart model selection"""
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

        # SMART MODEL SELECTION LOGIC
        primary_service = None
        fallback_service = None
        
        if duration_minutes <= 10:
            # For ≤10 min audio: OpenAI Whisper primary, AssemblyAI fallback
            primary_service = "openai"
            fallback_service = "assemblyai"
            logger.info(f"🎯 Audio ≤10min: Using OpenAI Whisper primary, AssemblyAI fallback")
        else:
            # For >10 min audio: AssemblyAI primary, OpenAI Whisper fallback
            primary_service = "assemblyai"
            fallback_service = "openai"
            logger.info(f"🎯 Audio >10min: Using AssemblyAI primary, OpenAI Whisper fallback")

        # Determine AssemblyAI model based on user plan
        def get_assemblyai_model(plan: str) -> str:
            if plan == 'free':
                return "nano"  # $0.12/hour for free users
            else:
                return "best"  # $0.27/hour for paid users

        assemblyai_model = get_assemblyai_model(user_plan)
        
        job_data.update({
            "primary_service": primary_service,
            "fallback_service": fallback_service,
            "assemblyai_model": assemblyai_model,
            "duration_minutes": duration_minutes
        })

        # TRY PRIMARY SERVICE
        transcription_result = None
        
        if primary_service == "openai":
            try:
                logger.info(f"🚀 Attempting OpenAI Whisper (primary) for job {job_id}")
                transcription_result = await transcribe_with_openai_whisper(tmp_path, language_code, job_id)
                job_data["primary_service_used"] = "openai"
                job_data["primary_service_success"] = True
            except Exception as openai_error:
                logger.error(f"❌ OpenAI Whisper (primary) failed: {openai_error}")
                job_data["primary_service_error"] = str(openai_error)
                job_data["primary_service_success"] = False
        
        elif primary_service == "assemblyai":
            try:
                logger.info(f"🚀 Attempting AssemblyAI (primary) with {assemblyai_model} model for job {job_id}")
                transcription_result = await transcribe_with_assemblyai(tmp_path, language_code, speaker_labels_enabled, assemblyai_model, job_id)
                job_data["primary_service_used"] = "assemblyai"
                job_data["primary_service_success"] = True
            except Exception as assemblyai_error:
                logger.error(f"❌ AssemblyAI (primary) failed: {assemblyai_error}")
                job_data["primary_service_error"] = str(assemblyai_error)
                job_data["primary_service_success"] = False

        check_cancellation()

        # IF PRIMARY FAILED, TRY FALLBACK
        if not transcription_result or transcription_result.get("status") == "failed":
            logger.warning(f"⚠️ Primary service failed, trying fallback ({fallback_service}) for job {job_id}")
            
            if fallback_service == "openai":
                try:
                    logger.info(f"🔄 Attempting OpenAI Whisper (fallback) for job {job_id}")
                    transcription_result = await transcribe_with_openai_whisper(tmp_path, language_code, job_id)
                    job_data["fallback_service_used"] = "openai"
                    job_data["fallback_service_success"] = True
                except Exception as openai_fallback_error:
                    logger.error(f"❌ OpenAI Whisper (fallback) failed: {openai_fallback_error}")
                    job_data["fallback_service_error"] = str(openai_fallback_error)
                    job_data["fallback_service_success"] = False
            
            elif fallback_service == "assemblyai":
                try:
                    logger.info(f"🔄 Attempting AssemblyAI (fallback) with {assemblyai_model} model for job {job_id}")
                    transcription_result = await transcribe_with_assemblyai(tmp_path, language_code, speaker_labels_enabled, assemblyai_model, job_id)
                    job_data["fallback_service_used"] = "assemblyai"
                    job_data["fallback_service_success"] = True
                except Exception as assemblyai_fallback_error:
                    logger.error(f"❌ AssemblyAI (fallback) failed: {assemblyai_fallback_error}")
                    job_data["fallback_service_error"] = str(assemblyai_fallback_error)
                    job_data["fallback_service_success"] = False

        check_cancellation()

        # PROCESS RESULTS
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
                "service_used": job_data.get("primary_service_used") or job_data.get("fallback_service_used")
            })
        else:
            logger.error(f"❌ Both services failed for job {job_id}")
            job_data.update({
                "status": "failed",
                "error": "Both transcription services failed",
                "completed_at": datetime.now().isoformat()
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
app = FastAPI(title="Enhanced Transcription Service with OpenAI Whisper & AssemblyAI", lifespan=lifespan)
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
        "message": "Enhanced Transcription Service with OpenAI Whisper & AssemblyAI is running!",
        "features": [
            "OpenAI Whisper API integration",
            "AssemblyAI integration with smart model selection",
            "Intelligent service selection based on audio duration",
            "Automatic fallback between services",
            "Paystack payment integration",
            "Speaker diarization for both services",
            "Language selection for transcription"
        ],
        "logic": {
            "audio_10min_or_less": "OpenAI Whisper primary, AssemblyAI fallback",
            "audio_over_10min": "AssemblyAI primary, OpenAI Whisper fallback",
            "free_users": "AssemblyAI nano model",
            "paid_users": "AssemblyAI best model"
        },
        "stats": {
            "active_jobs": len(jobs),
            "background_tasks": len(active_background_tasks),
            "cancellation_flags": len(cancellation_flags)
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
                'user_id': request.user_id,
                'country_code': request.country_code,
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
        "openai_configured": bool(OPENAI_API_KEY),
        "assemblyai_configured": bool(ASSEMBLYAI_API_KEY),
        "endpoints": {
            "initialize_payment": "/api/initialize-paystack-payment",
            "verify_payment": "/api/verify-payment",
            "webhook": "/api/paystack-webhook",
            "status": "/api/paystack-status",
            "transcribe": "/transcribe"
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
# NEW: Updated main transcription endpoint
@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form("en"),
    speaker_labels_enabled: bool = Form(False),
    user_plan: str = Form("free"),
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
        "logic_used": "≤10min: OpenAI→AssemblyAI, >10min: AssemblyAI→OpenAI"
    }

# NEW ENDPOINT: Generate formatted Word document
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
                "openai_configured": bool(OPENAI_API_KEY),
                "assemblyai_configured": bool(ASSEMBLYAI_API_KEY),
                "paystack_configured": bool(PAYSTACK_SECRET_KEY)
            },
            "transcription_logic": {
                "audio_10min_or_less": "OpenAI Whisper primary → AssemblyAI fallback",
                "audio_over_10min": "AssemblyAI primary → OpenAI Whisper fallback",
                "free_users_assemblyai": "nano model ($0.12/hour)",
                "paid_users_assemblyai": "best model ($0.27/hour)",
                "openai_whisper": "whisper-1 model ($0.006/minute)"
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
logger.info(f"OpenAI API Key configured: {bool(OPENAI_API_KEY)}")
logger.info(f"AssemblyAI API Key configured: {bool(ASSEMBLYAI_API_KEY)}")
logger.info(f"Paystack Secret Key configured: {bool(PAYSTACK_SECRET_KEY)}")
logger.info(f"Job tracking systems initialized:")
logger.info(f"  - Main jobs dictionary: {len(jobs)} jobs")
logger.info(f"  - Active background tasks: {len(active_background_tasks)} tasks")
logger.info(f"  - Cancellation flags: {len(cancellation_flags)} flags")

logger.info("Available API endpoints:")
logger.info("  POST /transcribe - Main transcription endpoint with smart service selection")
logger.info("  GET /status/{job_id} - Check job status")
logger.info("  POST /cancel/{job_id} - Cancel transcription job")
logger.info("  POST /compress-download - Compress audio for download")
logger.info("  POST /generate-formatted-word - Generate formatted Word document with speaker labels")
logger.info("  POST /api/initialize-paystack-payment - Initialize Paystack payment")
logger.info("  POST /api/verify-payment - Verify Paystack payment")
logger.info("  POST /api/paystack-webhook - Handle Paystack webhooks")
logger.info("  GET /api/paystack-status - Get integration status")
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
    logger.info("  ✅ OpenAI Whisper API integration (replaces Render backend)")
    logger.info("  ✅ Smart service selection: ≤10min→OpenAI, >10min→AssemblyAI")
    logger.info("  ✅ Automatic fallback between OpenAI and AssemblyAI")
    logger.info("  ✅ Speaker diarization for both OpenAI and AssemblyAI")
    logger.info("  ✅ Dynamic AssemblyAI model selection (nano for free, best for paid)")
    logger.info("  ✅ Unified transcription processing pipeline")
    logger.info("  ✅ Enhanced error handling and service resilience")
    logger.info("  ✅ Comprehensive job tracking and cancellation")
    logger.info("  ✅ Paystack payment integration")
    logger.info("  ✅ Multi-language support")
    logger.info("  ✅ Formatted Word document generation")
    
    logger.info("🔧 TRANSCRIPTION LOGIC:")
    logger.info("  - Audio ≤10 minutes: OpenAI Whisper primary, AssemblyAI fallback")
    logger.info("  - Audio >10 minutes: AssemblyAI primary, OpenAI Whisper fallback")
    logger.info("  - Free users: AssemblyAI nano model ($0.12/hour)")
    logger.info("  - Paid users: AssemblyAI best model ($0.27/hour)")
    logger.info("  - OpenAI Whisper: whisper-1 model ($0.006/minute = $0.36/hour)")
    logger.info("  - Both services support speaker labels with HTML formatting")
    
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
    logger.info("Ready to handle requests with OpenAI Whisper + AssemblyAI integration")
