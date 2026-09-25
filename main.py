import logging
import sys
import asyncio
import subprocess
import os
import json
import base64
import hashlib
import hmac
import calendar
from html import escape
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Response, Request, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uuid
import secrets
from datetime import datetime, timedelta
import requests
from pydub import AudioSegment

# Audio written for clients must open in ordinary playback software, not just
# in a browser. MP3 has three variants and the encoder picks one from the
# sample rate: 32 kHz and above gives MPEG-1, which everything plays; 16 to
# 24 kHz gives MPEG-2; and 8 to 12 kHz gives MPEG-2.5, a non-standard
# extension that transcription software such as ExpressScribe will not open.
# Staying at 44,100 Hz keeps every file we produce in the variant that works
# everywhere. Mono at 64 kbps is what keeps it small.
PLAYABLE_SAMPLE_RATE = 44100
PLAYABLE_BITRATE = "64k"
from pydantic import BaseModel
from typing import Optional, List
import httpx
from docx import Document
from docx.shared import Inches
from io import BytesIO
from fastapi.responses import StreamingResponse
import re
import anthropic

def claude_text(message) -> str:
    """Pull the answer text out of a Claude reply.

    Claude does not always put the answer in the first block. When a model
    reasons before answering, the first block is a thinking block, which has
    no .text at all, and reading it blindly raised
    "'ThinkingBlock' object has no attribute 'text'" in the client's face.
    Newer models can also return tool or citation blocks. So walk every
    block, keep the ones that actually carry text, and join them.
    """
    blocks = getattr(message, "content", None) or []
    parts = []
    for block in blocks:
        # Skip the reasoning; clients asked a question, not for the workings.
        if getattr(block, "type", None) in ("thinking", "redacted_thinking"):
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text)
    if parts:
        return "\n\n".join(parts).strip()
    logger.warning(
        "Claude reply carried no text blocks; types were %s",
        [getattr(b, "type", "?") for b in blocks],
    )
    return ""

import openai

import google.generativeai as genai

import firebase_admin
from firebase_admin import auth as firebase_auth, credentials, firestore, initialize_app, storage as firebase_storage
from google.cloud.firestore_v1.base_query import FieldFilter


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== STARTING FASTAPI APPLICATION (MAIN BACKEND) ===")

# Service Names
TYPEMYWORDZ1_NAME = "TypeMyworDz1" # AssemblyAI
TYPEMYWORDZ2_NAME = "TypeMyworDz2" # OpenAI Whisper
DEEPGRAM_NAME = "Deepgram" # Deepgram
TYPEMYWORDZ_AI_NAME = "TypeMyworDz AI" # Anthropic Claude / OpenAI GPT / Google Gemini

# Admin email addresses
# Configured admin accounts are never asked to subscribe, are never limited by
# plan or free-trial rules, and can reach the admin tools.
# Complimentary accounts (below) also skip payment, but are NOT admins and get
# none of the admin tooling.
ADMIN_EMAILS = ['typemywordz@gmail.com']
# Dedicated OpenAI Whisper tester. This account used to be the AssemblyAI
# tester; the owner moved it to OpenAI so OpenAI can be exercised on its own.
# Like the Deepgram tester it never falls back, so an OpenAI failure shows up
# in testing instead of being quietly masked by another provider. It is
# deliberately NOT a complimentary account: it must hold a plan or credits
# like any paying client, so it exercises the real billing path too.
OPENAI_TESTER_EMAIL = 'njokigituku@gmail.com'
# Dedicated Deepgram test account. It pays through the normal plan/credit
# path, starts on Deepgram, and falls back to OpenAI so both services can be
# tested without changing the experience for ordinary paying clients.
DEEPGRAM_TESTER_EMAIL = 'info@typemywordz.ai'

# Complimentary accounts. Keep this list empty unless an account is explicitly
# approved for free access. The Deepgram tester is intentionally a normal
# paying account so it exercises the real billing and paywall path.
COMP_ACCESS_EMAILS = []


def is_comp_access_user(user_email: str) -> bool:
    """Free to use the app, but not an admin."""
    if not user_email:
        return False
    return user_email.strip().lower() in [e.lower() for e in COMP_ACCESS_EMAILS]


# Outgoing email (Resend). The API key lives only in the environment, never in
# the repository. If it is missing the app still works: welcome emails are
# simply skipped and logged, rather than breaking anyone's signup.
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_ENDPOINT = "https://api.resend.com/emails"
# The sending subdomain is verified with Resend. It is deliberately a
# subdomain so that Resend's bounce handling cannot collide with the Zoho
# mailbox that receives mail on the root domain.
EMAIL_FROM = "TypeMyworDz <noreply@send.typemywordz.ai>"
SUPPORT_EMAIL = "info@typemywordz.ai"
APP_URL = "https://typemywordz.ai"

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

ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PAYSTACK_SECRET_KEY = os.environ.get("PAYSTACK_SECRET_KEY")
PAYSTACK_PUBLIC_KEY = os.environ.get("PAYSTACK_PUBLIC_KEY")
PAYSTACK_WEBHOOK_SECRET = os.environ.get("PAYSTACK_WEBHOOK_SECRET")
KORA_SECRET_KEY = os.environ.get("KORA_SECRET_KEY", "")
KORA_NOTIFICATION_URL = os.environ.get("KORA_NOTIFICATION_URL", "")
BACKEND_PUBLIC_URL = os.environ.get("BACKEND_PUBLIC_URL", "https://backendforrailway-production-7128.up.railway.app")
PADDLE_CLIENT_TOKEN = os.environ.get("PADDLE_CLIENT_TOKEN", "")
PADDLE_API_KEY = os.environ.get("PADDLE_API_KEY", "")
PADDLE_WEBHOOK_SECRET = os.environ.get("PADDLE_WEBHOOK_SECRET", "")
# Catalog product used for Paddle's server-created custom-price transactions.
# The product ID is public; the API key remains server-side.
PADDLE_CUSTOM_TOPUP_PRODUCT_ID = os.environ.get(
    "PADDLE_CUSTOM_TOPUP_PRODUCT_ID", "pro_01m2j2k9f0wrqz53j52z5rhr97"
)
try:
    PADDLE_PRICE_IDS = json.loads(os.environ.get("PADDLE_PRICE_IDS_JSON", "{}"))
except json.JSONDecodeError:
    PADDLE_PRICE_IDS = {}
PADDLE_PRICE_TO_ITEM = {str(price_id): str(item_id) for item_id, price_id in PADDLE_PRICE_IDS.items()}
OPENAI_WHISPER_SERVICE_RAILWAY_URL = os.environ.get("OPENAI_WHISPER_SERVICE_RAILWAY_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
FIREBASE_ADMIN_SDK_CONFIG_BASE64 = os.environ.get("FIREBASE_ADMIN_SDK_CONFIG_BASE64")
# Railway stores the Firebase Storage bucket under GCS_BUCKET_NAME. Accept both
# names so human-workflow uploads use the same configured bucket as the rest of
# the application.
FIREBASE_STORAGE_BUCKET = (os.environ.get("FIREBASE_STORAGE_BUCKET") or os.environ.get("GCS_BUCKET_NAME") or "").strip()
DEEPGRAM_SERVICE_RAILWAY_URL = os.environ.get("DEEPGRAM_SERVICE_RAILWAY_URL")

logger.info(f"DEBUG: --- Environment Variable Check (main.py) ---")
logger.info(f"DEBUG: ASSEMBLYAI_API_KEY loaded value: {bool(ASSEMBLYAI_API_KEY)}")
logger.info(f"DEBUG: ANTHROPIC_API_KEY loaded value: {bool(ANTHROPIC_API_KEY)}")
logger.info(f"DEBUG: OPENAI_API_KEY (for GPT if direct) loaded value: {bool(OPENAI_API_KEY)}")
logger.info(f"DEBUG: PAYSTACK_SECRET_KEY loaded value: {bool(PAYSTACK_SECRET_KEY)}")
logger.info(f"DEBUG: PAYSTACK_PUBLIC_KEY loaded value: {bool(PAYSTACK_PUBLIC_KEY)}")
logger.info(f"DEBUG: PAYSTACK_WEBHOOK_SECRET loaded value: {bool(PAYSTACK_WEBHOOK_SECRET)}")
logger.info(f"DEBUG: KORA_SECRET_KEY loaded value: {bool(KORA_SECRET_KEY)}")
logger.info(f"DEBUG: PADDLE_CLIENT_TOKEN loaded value: {bool(PADDLE_CLIENT_TOKEN)}")
logger.info(f"DEBUG: PADDLE_API_KEY loaded value: {bool(PADDLE_API_KEY)}")
logger.info(f"DEBUG: PADDLE_WEBHOOK_SECRET loaded value: {bool(PADDLE_WEBHOOK_SECRET)}")
logger.info(f"DEBUG: PADDLE_PRICE_IDS loaded count: {len(PADDLE_PRICE_IDS)}")
logger.info(f"DEBUG: OPENAI_WHISPER_SERVICE_RAILWAY_URL loaded value: {bool(OPENAI_WHISPER_SERVICE_RAILWAY_URL)}")
logger.info(f"DEBUG: GEMINI_API_KEY loaded value: {bool(GEMINI_API_KEY)}")
logger.info(f"DEBUG: FIREBASE_ADMIN_SDK_CONFIG_BASE64 loaded value: {bool(FIREBASE_ADMIN_SDK_CONFIG_BASE64)}")
logger.info(f"DEBUG: DEEPGRAM_SERVICE_RAILWAY_URL loaded value: {bool(DEEPGRAM_SERVICE_RAILWAY_URL)}")
logger.info(f"DEBUG: Admin emails configured: {ADMIN_EMAILS}")
logger.info(f"DEBUG: OpenAI Tester email: {OPENAI_TESTER_EMAIL}")
logger.info(f"DEBUG: --- End Environment Variable Check (main.py) ---")

if not ASSEMBLYAI_API_KEY:
    logger.error(f"{TYPEMYWORDZ1_NAME} API Key environment variable not set! {TYPEMYWORDZ1_NAME} will not work as primary or fallback.")

if not ANTHROPIC_API_KEY:
    logger.warning(f"{TYPEMYWORDZ_AI_NAME} (Anthropic) API Key environment variable not set! Anthropic AI features will be disabled.")

if not OPENAI_API_KEY:
    logger.warning(f"OPENAI_API_KEY (for GPT if direct) environment variable not set! Direct OpenAI GPT calls disabled.")

if not OPENAI_WHISPER_SERVICE_RAILWAY_URL:
    logger.error(f"{TYPEMYWORDZ2_NAME} (OpenAI Whisper & GPT) Service URL not configured! OpenAI transcription and GPT formatting will be disabled.")

if not GEMINI_API_KEY:
    logger.warning("Google Gemini API Key environment variable not set! Google Gemini AI features will be disabled.")

if not DEEPGRAM_SERVICE_RAILWAY_URL:
    logger.error(f"{DEEPGRAM_NAME} Service URL not configured! {DEEPGRAM_NAME} will not work as primary or fallback.")

if not FIREBASE_ADMIN_SDK_CONFIG_BASE64:
    logger.error("FIREBASE_ADMIN_SDK_CONFIG_BASE64 environment variable not set! Firebase Admin SDK features (user/revenue updates) will be disabled.")
else:
    try:
        decoded_json_test = base64.b64decode(FIREBASE_ADMIN_SDK_CONFIG_BASE64).decode('utf-8')
        parsed_json_test = json.loads(decoded_json_test)
        logger.info(f"DIAGNOSTIC (Runtime): Firebase config decoded and parsed successfully. Project ID: {parsed_json_test.get('project_id', 'N/A')}, Client Email: {parsed_json_test.get('client_email', 'N/A')}")
    except Exception as e:
        logger.error(f"DIAGNOSTIC (Runtime): ERROR: Failed to decode/parse FIREBASE_ADMIN_SDK_CONFIG_BASE64 at runtime: {e}")


if not PAYSTACK_SECRET_KEY:
    logger.warning("PAYSTACK_SECRET_KEY environment variable not set! Paystack features will be disabled.")

if PAYSTACK_SECRET_KEY:
    logger.info("Paystack configuration found - payment verification enabled")
else:
    logger.warning("Paystack configuration missing - payment verification disabled")

if PADDLE_CLIENT_TOKEN and PADDLE_PRICE_IDS:
    logger.info("Paddle checkout configuration found")
else:
    logger.warning("Paddle checkout configuration incomplete - global checkout disabled")
if not PADDLE_WEBHOOK_SECRET:
    logger.warning("PADDLE_WEBHOOK_SECRET not set - Paddle fulfillment webhook disabled")

logger.info("Environment variables loaded successfully")

# NEW: Initialize Firebase Admin SDK
db = None
if FIREBASE_ADMIN_SDK_CONFIG_BASE64:
    try:
        service_account_info = json.loads(base64.b64decode(FIREBASE_ADMIN_SDK_CONFIG_BASE64).decode('utf-8'))
        cred = credentials.Certificate(service_account_info)
        firebase_options = {"storageBucket": FIREBASE_STORAGE_BUCKET} if FIREBASE_STORAGE_BUCKET else {}
        initialize_app(cred, firebase_options)
        db = firestore.client()
        logger.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin SDK: {e}")
else:
    logger.warning("Firebase Admin SDK config is missing, Firestore operations will not be available.")


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return ""
    return header.split(" ", 1)[1].strip()


def _verified_user(request: Request) -> dict:
    token = _bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Sign-in is required.")
    try:
        return firebase_auth.verify_id_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Your sign-in session has expired.")


def _require_admin(request: Request) -> dict:
    decoded = _verified_user(request)
    email = (decoded.get("email") or "").strip().lower()
    if email not in {item.lower() for item in ADMIN_EMAILS}:
        raise HTTPException(status_code=403, detail="Admin access is required.")
    return decoded


def _read_admin_users_snapshot():
    if not db:
        return []
    profiles = []
    by_uid = {}
    for document in db.collection("users").stream():
        data = document.to_dict() or {}
        uid = data.get("uid") or document.id
        data["id"] = document.id
        data["uid"] = uid
        data["totalMinutesTranscribedByUser"] = 0
        data["totalTranscriptsByUser"] = 0
        data["askTypeMyworDzCreditsUsed"] = _int(data.get("askTypeMyworDzCreditsUsed"))
        data["askTypeMyworDzQueries"] = _int(data.get("askTypeMyworDzQueries"))
        data["transcriptAiCreditsUsed"] = _int(data.get("transcriptAiCreditsUsed"))
        data["transcriptAiQueries"] = _int(data.get("transcriptAiQueries"))
        profiles.append(data)
        by_uid[uid] = data

    for document in db.collection("transcriptions").stream():
        data = document.to_dict() or {}
        uid = data.get("userId")
        if not uid or uid not in by_uid:
            continue
        seconds = data.get("duration")
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            seconds = 0
        if seconds == seconds and seconds > 0 and seconds != float("inf") and seconds != float("-inf"):
            by_uid[uid]["totalMinutesTranscribedByUser"] += int((seconds + 59) // 60)
        by_uid[uid]["totalTranscriptsByUser"] += 1
    return profiles


def _delete_matching_documents(collection_name: str, field_name: str, value: str) -> int:
    if not db:
        return 0
    query_ref = db.collection(collection_name).where(
        filter=FieldFilter(field_name, "==", value)
    )
    snapshots = list(query_ref.stream())
    for snapshot in snapshots:
        snapshot.reference.delete()
    return len(snapshots)


claude_client = None
if ANTHROPIC_API_KEY:
    try:
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info(f"{TYPEMYWORDZ_AI_NAME} (Anthropic) client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing {TYPEMYWORDZ_AI_NAME} (Anthropic) client: {e}")
else:
    logger.warning(f"{TYPEMYWORDZ_AI_NAME} (Anthropic) API key is missing, Claude client will not be initialized.")

# Google Gemini Client initialization
gemini_client = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel('models/gemini-pro-latest')
        logger.info(f"Google Gemini client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Google Gemini client: {e}")
else:
    logger.warning(f"Google Gemini API key is missing, client will not be initialized.")


# ---------------------------------------------------------------------------
# Ask TypeMyworDz model catalogue.
#
# Every id below has been called successfully on this account. The page sends
# the id the client picked, but the SERVER decides whether that id is allowed
# for their plan, so nobody can unlock a premium model by editing the page.
#
# "standard" models come with any paid plan. "premium" models are part of what
# the Monthly and Yearly plans are for.
# ---------------------------------------------------------------------------

ASK_MODEL_CATALOGUE = [
    # -- standard: included with any paid plan ------------------------------
    {
        "id": "gpt-5.6-luna",
        "provider": "openai",
        "label": "ChatGPT 5.6 Luna",
        "blurb": "Fast, sharp, and reads very long transcripts. The best all-round choice.",
        "tier": "standard",
        "credits": 1,
        "transcript_only": False,
    },
    # Mistral is wired up and working, but their free tier rejects calls from
    # this server's address with "Invalid API Key" even though the same key
    # succeeds elsewhere. Until a payment method is added to the Mistral
    # account, the entry stays out of the catalogue so no client is offered a
    # model that cannot answer. To switch it back on, restore this block:
    #   {"id": "mistral-small-latest", "provider": "mistral",
    #    "label": "Mistral Small 4",
    #    "blurb": "A capable European model. Good for summaries and everyday questions.",
    #    "tier": "standard", "credits": 1, "transcript_only": False},
    {
        "id": "claude-haiku-4-5-20251001",
        "provider": "claude",
        "label": "Claude Haiku 4.5",
        "blurb": "Claude's quick model. Strong at careful reading and quoting.",
        "tier": "standard",
        "credits": 5,
        "transcript_only": False,
    },
    {
        "id": "gemini-3.1-flash-lite",
        "provider": "gemini",
        "label": "Gemini 3.1 Flash-Lite",
        "blurb": "Google's quick model. Available when you are working on a transcript.",
        "tier": "standard",
        "credits": 2,
        "transcript_only": True,
    },
    # -- premium: what the Monthly and Yearly plans are for -----------------
    {
        "id": "gpt-5.6-terra",
        "provider": "openai",
        "label": "ChatGPT 5.6 Terra",
        "blurb": "A step up in reasoning. Good for long or complicated material.",
        "tier": "premium",
        "credits": 10,
        "transcript_only": False,
    },
    {
        "id": "claude-sonnet-5",
        "provider": "claude",
        "label": "Claude Sonnet 5",
        "blurb": "An excellent all-rounder. Careful, thorough answers.",
        "tier": "premium",
        "credits": 10,
        "transcript_only": False,
    },
    {
        "id": "claude-opus-4-6",
        "provider": "claude",
        "label": "Claude Opus 4.6",
        "blurb": "The most capable Claude. Slower, best for difficult work.",
        "tier": "premium",
        "credits": 22,
        "transcript_only": False,
    },
    {
        "id": "gpt-5.6-sol",
        "provider": "openai",
        "label": "ChatGPT 5.6 Sol",
        "blurb": "OpenAI's most capable model. For the hardest questions.",
        "tier": "premium",
        "credits": 18,
        "transcript_only": False,
    },
    {
        "id": "gemini-3.6-flash",
        "provider": "gemini",
        "label": "Gemini 3.6 Flash",
        "blurb": "Google's deeper model. Available when you are working on a transcript.",
        "tier": "premium",
        "credits": 10,
        "transcript_only": True,
    },
]

ASK_DEFAULT_MODEL = "gpt-5.6-luna"

# A safe fallback for each provider, used if the default is somehow missing.
ASK_FALLBACK_MODEL = "claude-haiku-4-5-20251001"

# Plans that also get the premium models.
PREMIUM_AI_PLANS = ['Monthly Plan', 'Yearly Plan']

# Gemini's 3.x models spend part of their output budget "thinking", and those
# thinking tokens are billed and counted against max_output_tokens. With a
# small budget the visible answer gets cut off mid-sentence, which is exactly
# what clients were seeing. Give Gemini plenty of room, and switch thinking
# off on the models that allow it.
GEMINI_MIN_OUTPUT_TOKENS = 8000
GEMINI_THINKING_OFF = {"gemini-3.1-flash-lite"}


def ask_models_for(user_plan: str, user_email: str = "", has_transcript: bool = True, has_credits: bool = False):
    """Which models may this caller choose from?

    Admins get everything. Monthly and Yearly get the premium models on top of
    the standard ones. Every other paid plan gets the standard set.

    Some models are marked transcript_only. Those are offered when the question
    is about a transcript, but not on the standalone research page.
    """
    if not is_ai_allowed(user_plan, user_email, has_credits):
        return []
    premium_ok = (
        is_admin_user(user_email)
        or is_comp_access_user(user_email)
        or (user_plan in PREMIUM_AI_PLANS)
    )
    out = []
    for m in ASK_MODEL_CATALOGUE:
        if m["tier"] != "standard" and not premium_ok:
            continue
        if m.get("transcript_only") and not has_transcript:
            continue
        out.append(m)
    return out


def ask_models_locked_for(user_plan: str, user_email: str = "", has_transcript: bool = True, has_credits: bool = False):
    """Which models are being withheld from this caller purely because of plan?

    The Settings page shows these dimmed with a lock, rather than hiding them,
    so a client can see what a better plan would give them.

    Someone with no assistant access at all sees the WHOLE catalogue locked,
    because they are exactly the person a plan would help. Only the
    transcript-only models are left out on the research page, since a plan
    would not make those appear there either. This never affects what the
    server will accept.
    """
    ai_ok = is_ai_allowed(user_plan, user_email, has_credits)
    premium_ok = (
        is_admin_user(user_email)
        or is_comp_access_user(user_email)
        or (user_plan in PREMIUM_AI_PLANS)
    )
    if ai_ok and premium_ok:
        return []
    out = []
    for m in ASK_MODEL_CATALOGUE:
        # With no assistant access, everything is locked. With access but no
        # premium plan, only the premium ones are.
        if ai_ok and m["tier"] == "standard":
            continue
        if m.get("transcript_only") and not has_transcript:
            continue
        out.append(m)
    return out


def resolve_ask_model(requested: str, user_plan: str, user_email: str = "", has_transcript: bool = True, has_credits: bool = False):
    """Turn a requested model id into (model_id, provider), safely.

    An unknown id, or one the caller's plan does not include, quietly falls
    back to the default rather than failing. A client should never see an
    error because they had a stale model saved in their settings.
    """
    allowed = ask_models_for(user_plan, user_email, has_transcript, has_credits)
    if not allowed:
        return ASK_DEFAULT_MODEL, "openai"
    wanted = (requested or "").strip()
    for m in allowed:
        if m["id"] == wanted:
            return m["id"], m["provider"]
    # Older versions of the page sent a provider name rather than an id.
    if wanted in ("claude", "gemini", "openai", "mistral"):
        for m in allowed:
            if m["provider"] == wanted:
                return m["id"], m["provider"]
    for want in (ASK_DEFAULT_MODEL, ASK_FALLBACK_MODEL):
        for m in allowed:
            if m["id"] == want:
                return m["id"], m["provider"]
    return allowed[0]["id"], allowed[0]["provider"]


def ask_credit_cost(model_id: str) -> int:
    """How many credits one question on this model costs."""
    for m in ASK_MODEL_CATALOGUE:
        if m["id"] == model_id:
            return int(m.get("credits", 1))
    return 1


# ---------------------------------------------------------------------------
# Credit ledger
#
# One credit = one minute of transcription = one standard question.
#
# Every account has two purses:
#   planCredits   - included with a plan. They die when the plan dies.
#   topUpCredits  - bought separately. Valid for 12 months from purchase.
#
# Plan credits are always spent first, so that the credits with the nearer
# expiry go first and a client is never left holding plan credits that expire
# while their bought credits sit unused.
#
# The Yearly plan is "1,400 credits a month for a year" rather than 16,800 up
# front. That is handled by a refill date: whenever the balance is read after
# the refill date has passed, the monthly allowance is topped back up and the
# refill date moves on a month. Doing it lazily on read means there is no
# scheduled job to go wrong.
# ---------------------------------------------------------------------------

import math

# What each plan includes.
PLAN_CREDITS = {
    'One-Day Plan':   {'credits': 150,   'days': 1,   'monthly_refill': False},
    'Three-Day Plan': {'credits': 320,   'days': 3,   'monthly_refill': False},
    'One-Week Plan':  {'credits': 600,   'days': 7,   'monthly_refill': False},
    'Monthly Plan':   {'credits': 1400,  'days': 30,  'monthly_refill': False},
    'Yearly Plan':    {'credits': 1400,  'days': 365, 'monthly_refill': True},
}

FREE_TRIAL_CREDITS = 30          # once per account
TOPUP_VALID_DAYS = 365           # bought credits last a year
REFILL_DAYS = 30                 # yearly plan tops up every 30 days

# Top-up bundles. Prices are set on the pricing page; the server only needs to
# know how many credits each bundle id is worth, so that a client cannot ask
# for a bundle and be given someone else's credit count.
TOPUP_BUNDLES = {
    'topup-300':  300,
    'topup-800':  800,
    'topup-2000': 2000,
}
CUSTOM_TOPUP_MIN_AFRICA = 50
CUSTOM_TOPUP_MIN_GLOBAL = 100
CUSTOM_TOPUP_MAX = 50000
CUSTOM_TOPUP_RATE = {'africa': 0.01, 'global': 0.0133333333}
TRAINEE_PRODUCT = 'trainee-training'
# Training enrollment test price; keep checkout and displayed pricing aligned.
TRAINEE_PRICE_USD = 20.00
TRAINEE_COUNTRY = 'KE'

# What everything costs, in US dollars, decided here and nowhere else.
#
# There are two price lists. Which one a client is offered depends on where
# they are paying from. The pricing page never announces this; it simply shows
# one set of prices, and the client's browser cannot change them. The browser
# says which plan or bundle it wants and the price is looked up here, so a
# tampered request cannot buy a Yearly plan for a dollar.
PRICES = {
    'africa': {
        'One-Day Plan':    1.50,
        'Three-Day Plan':  3.00,
        'One-Week Plan':   5.00,
        'Monthly Plan':    9.00,
        'Yearly Plan':    90.00,
        'topup-300':       3.00,
        'topup-800':       6.50,
        'topup-2000':     15.00,
    },
    'global': {
        # No One-Day Plan on the global list.
        #
        # Our card processor for the rest of the world charges 5% plus a flat
        # 50 cents per transaction. On a $2.00 sale the flat fee alone is 25%
        # of the price, and after compute the plan cleared about 44 cents --
        # one support email or one refund wiped out several of them. The
        # cheapest global entry point is now the 300-credit bundle at $4.00,
        # which has no expiry pressure and clears about $2.10.
        #
        # African clients keep the $1.50 One-Day Plan, where the local
        # processor's fee structure makes it work.
        'Three-Day Plan':  4.00,
        'One-Week Plan':   7.00,
        'Monthly Plan':   14.00,
        'Yearly Plan':   140.00,
        'topup-300':       4.00,
        'topup-800':       9.00,
        'topup-2000':     20.00,
    },
}

AFRICA_PAYMENT_CODES = {'KE', 'NG', 'GH', 'ZA', 'OTHER_AFRICA'}


def price_region(country_code):
    """Which price list applies. Anything not recognised pays the global list."""
    return 'africa' if (country_code or '').upper() in AFRICA_PAYMENT_CODES else 'global'


def custom_topup_min(country_code='KE'):
    return CUSTOM_TOPUP_MIN_GLOBAL if price_region(country_code) == 'global' else CUSTOM_TOPUP_MIN_AFRICA


def custom_topup_credits(item, country_code='KE'):
    match = re.fullmatch(r'topup-custom-(\d+)', str(item or ''))
    if not match:
        return None
    credits = int(match.group(1))
    if credits < custom_topup_min(country_code) or credits > CUSTOM_TOPUP_MAX:
        return None
    return credits


def custom_topup_price(credits, country_code):
    region = price_region(country_code)
    return round(float(credits) * CUSTOM_TOPUP_RATE[region], 2)


def price_for(item, country_code):
    """The dollar price of a plan, bundle, custom top-up, or trainee enrollment."""
    if item == TRAINEE_PRODUCT:
        return TRAINEE_PRICE_USD if (country_code or '').upper() == TRAINEE_COUNTRY else None
    custom = custom_topup_credits(item, country_code)
    if custom is not None:
        return custom_topup_price(custom, country_code)
    return PRICES[price_region(country_code)].get(item)


def _as_dt(value):
    """Firestore hands back several date shapes. Normalise or give up safely."""
    if value is None:
        return None
    if isinstance(value, datetime):
        # Firestore's DatetimeWithNanoseconds can survive replace() as its
        # subclass, then fail when written back because Firestore expects its
        # internal nanosecond field. Rebuild as a plain datetime; this app uses
        # microsecond precision for deadlines and payout records.
        return datetime(value.year, value.month, value.day, value.hour, value.minute, value.second, value.microsecond)
    for attr in ('to_datetime', 'ToDatetime'):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return fn().replace(tzinfo=None)
            except Exception:
                return None
    try:
        return datetime.fromisoformat(str(value).replace('Z', '')).replace(tzinfo=None)
    except Exception:
        return None


def _int(value):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0



# TMWD_HUMAN_QUOTE_V1
# Human transcription is priced separately from AI transcription.
# One human job minute is deliberately not one AI credit: the credit bundles
# are priced for machine transcription, while a human transcriber is paid per
# completed audio minute. These values are the first explicit internal rule for
# the bridge and can be changed in one place before production charging starts.
HUMAN_STANDARD_CREDITS_PER_MINUTE = 40
HUMAN_RUSH_CREDITS_PER_MINUTE = 55
# Standard transcription pay is 25 KES per completed audio minute. Rush pay
# remains separately configured; proofreading stays at 10 KES per minute.
HUMAN_STANDARD_PAYOUT_KES = 25
HUMAN_RUSH_PAYOUT_KES = 38
HUMAN_PROOFREADING_PAYOUT_KES = 10

# TMWD_HUMAN_TAT_V2
# Jobs of one minute or less receive a six-minute minimum TAT. Longer jobs
# receive four minutes for every rounded audio minute. If the worker has not
# submitted by the deadline, the job is returned to the admin queue.
HUMAN_TAT_MINUTES_PER_AUDIO_MINUTE = 4
HUMAN_SHORT_AUDIO_TAT_MINUTES = 6
HUMAN_PROOFREADING_TAT_MINUTES_PER_AUDIO_MINUTE = 1.5
HUMAN_PROOFREADING_SHORT_AUDIO_MINUTES = 3
HUMAN_PROOFREADING_SHORT_TAT_MINUTES = 5
HUMAN_TAT_EXTENSION_OPTIONS = (5, 10, 15, 20)


def human_tat_seconds(audio_seconds):
    """Return the worker deadline window for one assigned audio segment."""
    try:
        seconds = float(audio_seconds or 0)
    except (TypeError, ValueError):
        seconds = 0
    if seconds <= 60:
        return HUMAN_SHORT_AUDIO_TAT_MINUTES * 60
    minutes = max(1, int(math.ceil(seconds / 60.0)))
    return minutes * HUMAN_TAT_MINUTES_PER_AUDIO_MINUTE * 60


def human_proofreading_tat_seconds(audio_seconds):
    """Proofreading gets a shorter TAT: five minutes through three audio
    minutes, then 1.5 minutes for each rounded-up audio minute."""
    try:
        seconds = float(audio_seconds or 0)
    except (TypeError, ValueError):
        seconds = 0
    if seconds <= 0:
        return HUMAN_PROOFREADING_SHORT_TAT_MINUTES * 60
    audio_minutes = max(1, int(math.ceil(seconds / 60.0)))
    if audio_minutes <= HUMAN_PROOFREADING_SHORT_AUDIO_MINUTES:
        return HUMAN_PROOFREADING_SHORT_TAT_MINUTES * 60
    return int(math.ceil(audio_minutes * HUMAN_PROOFREADING_TAT_MINUTES_PER_AUDIO_MINUTE * 60))


def human_credit_quote(seconds, turnaround="standard", difficulty="standard", service="standard", speakers="1-2", timestamps=True, formatting="standard"):
    """Return one server-owned quote for both human-work entry points."""
    try:
        duration = float(seconds or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Audio duration must be a finite, non-negative number.")

    rush = str(turnaround or "").strip().lower() == "rush"
    difficult = str(difficulty or "").strip().lower() == "difficult"
    premium = rush or difficult
    minutes = max(1, int(math.ceil(duration / 60.0)))
    rate = HUMAN_RUSH_CREDITS_PER_MINUTE if premium else HUMAN_STANDARD_CREDITS_PER_MINUTE
    payout = HUMAN_RUSH_PAYOUT_KES if premium else HUMAN_STANDARD_PAYOUT_KES
    multiplier = 1.0
    if str(service or "").strip().lower() == "proofread": multiplier *= 1.2
    if str(service or "").strip().lower() == "formatted": multiplier *= 1.3
    if str(speakers or "").strip().lower() == "3+": multiplier *= 1.1
    if timestamps is False or str(timestamps).lower() == "false": multiplier *= 0.95
    if str(formatting or "").strip().lower() == "advanced": multiplier *= 1.1
    credits = int(math.ceil(minutes * rate * multiplier))
    return {
        "minutes": minutes,
        "credits": credits,
        "credits_per_minute": rate,
        "transcriber_payout_kes_per_minute": payout,
        "pricing_tier": "rush_or_difficult" if premium else "standard",
        "service": service,
        "speakers": speakers,
        "timestamps": bool(timestamps) if not isinstance(timestamps, str) else timestamps.lower() != "false",
        "formatting": formatting,
        "formula_version": "human-v2",
    }


def minutes_to_credits(seconds):
    """Transcription is charged by the minute, rounded up, minimum one."""
    try:
        s = float(seconds or 0)
    except (TypeError, ValueError):
        s = 0.0
    if s <= 0:
        return 1
    return max(1, int(math.ceil(s / 60.0)))


def read_balance(profile, now=None):
    """What can this account actually spend right now?

    Pure: it reads a profile and returns numbers plus any writes that ought to
    be made (a yearly refill, or clearing an expired purse). It never writes.
    """
    now = now or datetime.now()
    profile = profile or {}
    updates = {}

    plan = profile.get('plan')
    plan_credits = _int(profile.get('planCredits'))
    plan_expires = _as_dt(profile.get('planCreditsExpireAt'))
    refill_at = _as_dt(profile.get('planCreditsRefillAt'))

    # Repair: a plan that was paid for but never filled the purse.
    #
    # Credits are meant to be granted at the moment a payment clears. For a
    # period they were not: the plan was written to the account and the purse
    # was left untouched, so a paying client saw a live plan and zero credits.
    #
    # Those accounts are recognisable because the plan purse has no expiry
    # date at all. That is genuinely different from a client who simply spent
    # everything, whose purse has a date and a zero against it, so this cannot
    # refill someone who has legitimately run out. The grant is made once, and
    # only for what is left of the plan they already paid for, so it can never
    # hand out more than was bought or extend the plan.
    if plan_expires is None:
        spec = PLAN_CREDITS.get(plan)
        plan_ends = _as_dt(profile.get('expiresAt'))
        if spec and plan_ends is not None and now < plan_ends:
            plan_credits = spec['credits']
            plan_expires = plan_ends
            updates['planCredits'] = plan_credits
            updates['planCreditsExpireAt'] = plan_ends
            if spec['monthly_refill']:
                refill_at = now + timedelta(days=REFILL_DAYS)
                updates['planCreditsRefillAt'] = refill_at

    # Plan credits die with the plan.
    if plan_expires is not None and now >= plan_expires:
        if plan_credits:
            updates['planCredits'] = 0
        plan_credits = 0
    else:
        # Yearly plan: top the monthly allowance back up, catching up if the
        # client has been away for several months.
        spec = PLAN_CREDITS.get(plan)
        if spec and spec['monthly_refill'] and refill_at is not None:
            moved = False
            guard = 0
            while now >= refill_at and guard < 24:
                plan_credits = spec['credits']
                refill_at = refill_at + timedelta(days=REFILL_DAYS)
                moved = True
                guard += 1
            if moved:
                updates['planCredits'] = plan_credits
                updates['planCreditsRefillAt'] = refill_at

    topup_credits = _int(profile.get('topUpCredits'))
    topup_expires = _as_dt(profile.get('topUpCreditsExpireAt'))
    if topup_expires is not None and now >= topup_expires:
        if topup_credits:
            updates['topUpCredits'] = 0
        topup_credits = 0

    # Two different purses, two different rules.
    #
    # Plan credits come WITH a plan, so they stop when the plan stops. That is
    # what the client bought and what the dates on the plan say.
    #
    # Top-up credits were paid for separately, in cash, on top of a plan. The
    # owner's decision is that this money stays spendable even after a plan
    # lapses, because freezing money somebody has already handed over is the
    # fastest way to earn a complaint.
    #
    # This is not a loophole. Measured against the real price list, a client
    # living on top-ups alone always pays MORE per credit than a subscriber:
    # the cheapest top-up works out at about 0.0070 per credit against 0.0049
    # on the cheapest plan, and every bundle clears cost with at least a 42
    # percent margin. Avoiding a subscription costs the client more, not less.
    plan_active = plan_expires is not None and now < plan_expires
    spendable = (plan_credits if plan_active else 0) + topup_credits

    return {
        'planCredits': plan_credits,
        'topUpCredits': topup_credits,
        'total': plan_credits + topup_credits,
        'planActive': plan_active,
        'spendable': spendable,
        # Nothing is frozen any more, and this is deliberate rather than an
        # oversight. Plan credits are already set to zero a few lines above
        # when the plan expires, because they belong to the plan. Bought
        # credits are now always spendable. That leaves nothing in between,
        # so this is always 0. The key is kept because the client reads it,
        # and keeping it means an older client build cannot break.
        'frozen': 0,
        'planCreditsExpireAt': plan_expires,
        'topUpCreditsExpireAt': topup_expires,
        'updates': updates,
    }


def plan_spend(profile, amount, now=None):
    """Work out how to take `amount` credits, plan purse first.

    Returns (ok, updates, detail). Does not write anything.
    """
    amount = _int(amount)
    bal = read_balance(profile, now)
    updates = dict(bal['updates'])

    if amount <= 0:
        return True, updates, {'charged': 0, 'from_plan': 0, 'from_topup': 0,
                               'remaining': bal['spendable']}

    if bal['spendable'] < amount:
        return False, updates, {'needed': amount, 'available': bal['spendable'],
                                'short_by': amount - bal['spendable'],
                                'frozen': bal['frozen'],
                                'planActive': bal['planActive'],
                                'reason': ('no active plan'
                                           if (not bal['planActive'] and bal['frozen'])
                                           else 'not enough credits')}

    # Spend the plan purse first, because it dies with the plan while bought
    # credits last a year. Only what is actually spendable counts.
    usable_plan = bal['planCredits'] if bal['planActive'] else 0
    from_plan = min(usable_plan, amount)
    from_topup = amount - from_plan

    if from_plan:
        updates['planCredits'] = bal['planCredits'] - from_plan
    if from_topup:
        updates['topUpCredits'] = bal['topUpCredits'] - from_topup

    return True, updates, {
        'charged': amount,
        'from_plan': from_plan,
        'from_topup': from_topup,
        'remaining': bal['spendable'] - amount,
    }


def grant_plan_credits(plan, now=None):
    """The writes that turn a successful payment into plan credits."""
    now = now or datetime.now()
    spec = PLAN_CREDITS.get(plan)
    if not spec:
        return {}
    out = {
        'planCredits': spec['credits'],
        'planCreditsExpireAt': now + timedelta(days=spec['days']),
    }
    out['planCreditsRefillAt'] = (
        now + timedelta(days=REFILL_DAYS) if spec['monthly_refill'] else None
    )
    return out


def grant_topup_credits(profile, bundle_id, country_code='KE', now=None):
    """Add a bought bundle. Bought credits stack and the clock restarts.

    Restarting the 12 months on every purchase is deliberately generous: it is
    easier to explain than several piles with different expiry dates, and it
    means a regular client's credits never quietly expire.
    """
    now = now or datetime.now()
    credits = TOPUP_BUNDLES.get(bundle_id)
    if credits is None:
        credits = custom_topup_credits(bundle_id, country_code)
    if not credits:
        return None
    bal = read_balance(profile, now)
    updates = dict(bal['updates'])
    updates['topUpCredits'] = bal['topUpCredits'] + credits
    updates['topUpCreditsExpireAt'] = now + timedelta(days=TOPUP_VALID_DAYS)
    return {'updates': updates, 'added': credits,
            'newTotal': bal['planCredits'] + updates['topUpCredits']}


def grant_free_trial(profile, now=None):
    """Ten credits, once, for a brand new account."""
    profile = profile or {}
    if profile.get('hasReceivedInitialFreeMinutes'):
        return {}
    now = now or datetime.now()
    return {
        'planCredits': FREE_TRIAL_CREDITS,
        'planCreditsExpireAt': now + timedelta(days=30),
        'planCreditsRefillAt': None,
        'hasReceivedInitialFreeMinutes': True,
    }


# --- end of credit ledger ---


# --- one-off conversion of the old hours system into credits ---
#
# Before credits, a plan included a number of transcription hours and the app
# counted minutes used. Those clients are mid-plan and must not lose anything,
# so their remaining minutes become credits one for one, and they are never
# given less than the new plan's own allowance. Eight hours left therefore
# becomes 480 credits rather than the 320 a fresh Three-Day plan grants.
#
# This runs once per account and records that it has run, so a client calling
# it twice does not get paid twice.

LEGACY_PLAN_MINUTES = {
    'One-Day Plan':   4 * 60,
    'Three-Day Plan': 8 * 60,
    'One-Week Plan':  15 * 60,
    'Monthly Plan':   25 * 60,
    'Yearly Plan':    25 * 60,
}


def backfill_credits(profile, now=None):
    """Turn a pre-credits profile into a credit balance. Pure; returns writes.

    Returns (updates, detail). An empty updates dict means there was nothing
    to do, which is the normal answer for everyone after the first call.
    """
    now = now or datetime.now()
    profile = profile or {}

    if profile.get('creditsBackfilledAt'):
        return {}, {'skipped': 'already done'}
    # An account that already has a live credit purse came in after credits
    # existed, so there is nothing to convert.
    if _as_dt(profile.get('planCreditsExpireAt')) is not None:
        return {'creditsBackfilledAt': now}, {'skipped': 'already on credits'}

    plan = profile.get('plan')
    spec = PLAN_CREDITS.get(plan)
    expires = _as_dt(profile.get('expiresAt'))

    if spec and expires is not None and expires > now:
        allowance = LEGACY_PLAN_MINUTES.get(plan, 0)
        used = _int(profile.get('totalMinutesUsed'))
        remaining = max(0, allowance - used)
        credits = max(remaining, spec['credits'])
        updates = {
            'planCredits': credits,
            'planCreditsExpireAt': expires,
            'planCreditsRefillAt': (now + timedelta(days=REFILL_DAYS)
                                    if spec['monthly_refill'] else None),
            'creditsBackfilledAt': now,
        }
        return updates, {'granted': credits, 'plan': plan,
                         'remainingMinutes': remaining, 'reason': 'paid plan converted'}

    # Free account that has not used its trial yet: give it the trial in
    # credits. One that has already used it gets nothing, which matches what
    # it had before.
    if not profile.get('hasReceivedInitialFreeMinutes'):
        updates = dict(grant_free_trial(profile, now))
        updates['creditsBackfilledAt'] = now
        return updates, {'granted': FREE_TRIAL_CREDITS, 'reason': 'free trial'}

    return {'creditsBackfilledAt': now}, {'granted': 0, 'reason': 'trial already used'}


# --- end of backfill ---


# --- referrals: give credits, get credits ---
#
# Every account gets a short code. Sharing it and having someone sign up
# tops up both people by the same amount, once per new account. This reuses
# the existing top-up ledger rather than inventing a new balance, so a
# referral bonus behaves exactly like a bought bundle: it stacks, and it
# lasts a year.

REFERRAL_BONUS_CREDITS = 100
REFERRAL_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I


def _new_referral_code() -> str:
    return "".join(secrets.choice(REFERRAL_CODE_ALPHABET) for _ in range(7))


async def _ensure_referral_code(user_id: str, profile: dict) -> str:
    """Return this account's referral code, creating one the first time."""
    existing = (profile or {}).get("referralCode")
    if existing:
        return existing
    for _ in range(5):
        code = _new_referral_code()
        doc_ref = db.collection("referral_codes").document(code)
        snap = await asyncio.to_thread(doc_ref.get)
        if snap.exists:
            continue
        await asyncio.to_thread(doc_ref.set, {"uid": user_id, "createdAt": datetime.now()})
        await asyncio.to_thread(db.collection("users").document(user_id).update, {"referralCode": code})
        return code
    raise HTTPException(status_code=500, detail="Could not generate a referral code right now. Try again shortly.")


def _grant_referral_bonus(profile: dict, now=None) -> dict:
    """Same shape as a bought top-up: add credits, extend the expiry."""
    now = now or datetime.now()
    bal = read_balance(profile or {}, now)
    updates = dict(bal["updates"])
    updates["topUpCredits"] = bal["topUpCredits"] + REFERRAL_BONUS_CREDITS
    updates["topUpCreditsExpireAt"] = now + timedelta(days=TOPUP_VALID_DAYS)
    return updates


# --- end of referrals ---


def is_paid_ai_user(user_plan: str) -> bool:
    paid_plans_for_ai = ['One-Day Plan', 'Three-Day Plan', 'One-Week Plan', 'Monthly Plan', 'Yearly Plan']
    return user_plan in paid_plans_for_ai


def is_ai_allowed(user_plan: str, user_email: str = "", has_credits: bool = False) -> bool:
    """May this caller use the AI features?

    Yes if they are on a paid plan, if they hold credits they can spend, if
    they are the admin, or if they are a complimentary account.

    Credits count exactly as a plan does. A client who buys credits has paid
    us the same money as a subscriber and must get the same features; being
    sold credits and then told the assistant is for subscribers only would be
    taking money for nothing. The admin and complimentary accounts keep plan
    "free" in the database and are recognised by email everywhere else in the
    app, so the AI endpoints must do the same or they are locked out.
    """
    if is_admin_user(user_email):
        return True
    if is_comp_access_user(user_email):
        return True
    if has_credits:
        return True
    return is_paid_ai_user(user_plan)

def credits_exempt(user_email: str) -> bool:
    """Accounts that use the app without spending credits.

    The admin and any complimentary account. Everyone else pays their way.
    """
    return is_admin_user(user_email) or is_comp_access_user(user_email)


# Extra accounts that can run the human-transcription pipeline solo: approve a
# request, assign it straight to a worker, and sign off finished work on a
# client's behalf, all without a separate client account clicking anything.
# This is deliberately its own list, kept separate from ADMIN_EMAILS, so it
# never grants the app-wide admin dashboard, never bypasses the paywall for
# AI transcription or Ask TypeMyworDz, and never exempts anything except the
# human-transcription jobs these accounts personally own as the "client".
HUMAN_JOB_ADMIN_EMAILS = ['info@typemywordz.ai']


def is_human_job_admin(user_email: str) -> bool:
    """True for a real admin, or one of the extra accounts above, for the
    purposes of the human-transcription workflow only."""
    if not user_email:
        return False
    email = user_email.strip().lower()
    if is_admin_user(email):
        return True
    return email in {item.lower() for item in HUMAN_JOB_ADMIN_EMAILS}


def human_job_credits_exempt(user_email: str) -> bool:
    """A human-transcription job is free of charge only when the client on
    that job is a real admin or one of the extra human-job-admin accounts.
    Deliberately separate from credits_exempt: it must never leak into AI
    transcription or Ask TypeMyworDz billing for the same accounts, so those
    keep exercising the real paywall exactly as before."""
    return is_human_job_admin(user_email)


def _require_human_job_admin(request: Request) -> dict:
    decoded = _verified_user(request)
    email = (decoded.get("email") or "").strip().lower()
    if not is_human_job_admin(email):
        raise HTTPException(status_code=403, detail="Admin access is required.")
    return decoded


async def _load_profile(user_id: str):
    if not db or not user_id:
        return None
    try:
        snap = await asyncio.to_thread(db.collection('users').document(user_id).get)
        return snap.to_dict() if snap.exists else None
    except Exception as e:
        logger.error(f"Could not read profile {user_id} for credits: {e}")
        return None


async def account_has_usable_credits(user_id: str = "", user_email: str = "") -> bool:
    """Does this account hold credits it can spend right now?

    Credits open the same doors a plan does, so every gate that used to ask
    only "are they subscribed?" has to ask this as well. A failure to read the
    account answers no, which is the safe direction: the client is told to buy
    something rather than being given something free.
    """
    try:
        if not user_id and user_email:
            user_id = await get_user_profile_by_email_firestore(user_email)
        if not user_id:
            return False
        profile = await _load_profile(user_id)
        if not profile:
            return False
        return read_balance(profile)['spendable'] > 0
    except Exception as e:
        logger.warning(f"Could not check credits for {user_email or user_id}: {e}")
        return False


async def _record_credit_ledger(user_id: str, amount: int, reason: str, context: Optional[dict] = None, balance_after: Optional[int] = None):
    """Write one immutable audit entry for every credit movement."""
    if not db or not user_id or not amount:
        return False
    payload = {
        "amount": int(amount),
        "direction": "added" if amount > 0 else "deducted",
        "reason": str(reason or "account credit update")[:240],
        "context": context if isinstance(context, dict) else {"detail": str(context or "")[:500]},
        "createdAt": firestore.SERVER_TIMESTAMP,
    }
    if balance_after is not None:
        payload["balanceAfter"] = int(balance_after)
    try:
        await asyncio.to_thread(
            db.collection("users").document(user_id).collection("credit_ledger").add,
            payload,
        )
        return True
    except Exception as e:
        logger.error("Could not write credit ledger entry for %s: %s", user_id, e)
        return False


async def _save_credit_updates(user_id: str, updates: dict, ledger_reason: str = "account credit update", ledger_context: Optional[dict] = None):
    if not db or not user_id or not updates:
        return False
    try:
        user_ref = db.collection("users").document(user_id)
        before_snapshot = await asyncio.to_thread(user_ref.get)
        before = before_snapshot.to_dict() if before_snapshot.exists else {}
        await asyncio.to_thread(user_ref.update, updates)
        before_total = _int(before.get("planCredits")) + _int(before.get("topUpCredits"))
        after_plan = _int(updates.get("planCredits", before.get("planCredits")))
        after_topup = _int(updates.get("topUpCredits", before.get("topUpCredits")))
        delta = (after_plan + after_topup) - before_total
        if delta:
            await _record_credit_ledger(
                user_id, delta, ledger_reason, ledger_context, after_plan + after_topup
            )
        return True
    except Exception as e:
        logger.error(f"Could not write credits for {user_id}: {e}")
        return False


async def charge_credits(user_id: str, user_email: str, amount: int, what: str, usage_category: str = ""):
    """Take credits for something the client has just received.

    Deliberately forgiving. If the ledger cannot be reached we log it and let
    the client keep what they have already been given, because silently losing
    a finished transcript over a database hiccup is far worse than missing one
    charge. The balance check that guards the paywall happens before the work
    starts, not here.
    """
    if credits_exempt(user_email):
        return {'charged': 0, 'exempt': True}
    # Transcription jobs are tracked by email, the assistant by id. Accept
    # either, and look the id up when only the email is to hand.
    if not user_id and user_email:
        user_id = await get_user_profile_by_email_firestore(user_email)
    if not user_id:
        return {'charged': 0, 'error': 'account not identified'}
    profile = await _load_profile(user_id)
    if profile is None:
        return {'charged': 0, 'error': 'profile unavailable'}
    ok, updates, detail = plan_spend(profile, amount)
    usage_fields = {
        "standalone_ask": ("askTypeMyworDzCreditsUsed", "askTypeMyworDzQueries"),
        "transcript_query": ("transcriptAiCreditsUsed", "transcriptAiQueries"),
    }
    if ok and usage_category in usage_fields and detail.get("charged", 0) > 0:
        credits_field, queries_field = usage_fields[usage_category]
        updates[credits_field] = firestore.Increment(int(detail["charged"]))
        updates[queries_field] = firestore.Increment(1)
    if updates:
        await _save_credit_updates(
            user_id,
            updates,
            ledger_reason=what,
            ledger_context={"operation": "charge", "usage_category": usage_category or None},
        )
    if ok:
        logger.info(f"Charged {detail.get('charged')} credits to {user_id} for {what}; {detail.get('remaining')} left")
    else:
        logger.warning(f"Could not charge {amount} credits to {user_id} for {what}: {detail}")
    return detail


def is_admin_user(user_email: str) -> bool:
    """Check if user is an admin based on email address"""
    if not user_email:
        return False
    return user_email.lower().strip() in [email.lower() for email in ADMIN_EMAILS]

def get_transcription_services(user_plan: str, speaker_labels_enabled: bool, user_email: str = None):
    """
    Logic for service selection based on new rules, including Deepgram.
    - OpenAI: First option for weekly subscribers, yearly, and Admins (Admins gets this logic no matter what plans they have subscribed to). Fallback is Assembly > Deepgram.
    - Assembly: First option for free users. Fallback Deepgram only (free users don't get TypeMyworDz Assistant) All instances of speaker tags requests: First option Deepgram, fallback Assembly.
    - Deepgram: First option for three-day and monthly plans users. Fallback is OpenAI > Assembly. All instances of speaker tags requests: First option Assembly, fallback Deepgram.
    - njokigituku@gmail.com is the dedicated OpenAI tester: OpenAI only, no fallback, and it pays like any client.
    - info@typemywordz.ai is the dedicated Deepgram tester: Deepgram first,
      OpenAI fallback, and normal plan/credit requirements.
    """
    
    is_admin = is_admin_user(user_email) if user_email else False
    is_openai_tester = (user_email and user_email.lower().strip() == OPENAI_TESTER_EMAIL.lower())
    is_deepgram_tester = (user_email and user_email.lower().strip() == DEEPGRAM_TESTER_EMAIL.lower())

    # --- Initialize tiers ---
    tier_1 = None
    tier_2 = None
    tier_3 = None
    reason = "default_logic"

    # --- Dedicated OpenAI Tester Logic ---
    # Placed before the speaker-label override on purpose: this account exists
    # to exercise OpenAI and nothing else, so even a speaker-tag request stays
    # on it rather than being handed to AssemblyAI.
    if is_openai_tester:
        return {
            "tier_1": "openai_whisper",
            "tier_2": None,
            "tier_3": None,
            "reason": "dedicated_openai_tester"
        }

    # --- Dedicated Deepgram Tester Logic ---
    # Placed before the speaker-label override on purpose: this account exists to
    # exercise Deepgram first, while still exposing the configured OpenAI
    # fallback if Deepgram is unavailable. It is not payment-exempt.
    if is_deepgram_tester:
        return {
            "tier_1": "deepgram",
            "tier_2": "openai_whisper",
            "tier_3": None,
            "reason": "dedicated_deepgram_tester_with_openai_fallback"
        }

    # --- Speaker Labels Logic (Global Override) ---
    # AssemblyAI first, because it diarizes well. OpenAI sits second and Deepgram
    # last. OpenAI does not return speaker tags, so a job that falls through to it
    # comes back as continuous text rather than empty: measured against a real
    # 8 kHz meeting recording, Deepgram returned 8 characters where AssemblyAI
    # returned 7,343, so it is no longer trusted as the first fallback.
    if speaker_labels_enabled:
        tier_1 = "assemblyai"
        tier_2 = "openai_whisper"
        tier_3 = "deepgram"
        reason = "speaker_labels_requested_prioritizing_assemblyai"

    # --- Plan-based logic ---
    # Note the elif: when speaker labels are requested the rule above wins
    # and none of these run. Between April and now this was an "if", which
    # quietly overrode the speaker-label rule for admins and yearly users.

    # AssemblyAI first for admins, One-Day, Three-Day and One-Week plans.
    elif is_admin or user_plan in ['One-Day Plan', 'Three-Day Plan', 'One-Week Plan']:
        tier_1 = "assemblyai"
        tier_2 = "openai_whisper"
        tier_3 = "deepgram"
        reason = "admin_or_short_plan_prioritizing_assemblyai"

    # Yearly is the most expensive plan, so it gets the same order as admins.
    elif user_plan == 'Yearly Plan':
        tier_1 = "assemblyai"
        tier_2 = "openai_whisper"
        tier_3 = "deepgram"
        reason = "yearly_plan_prioritizing_assemblyai"

    # Monthly subscribers start on OpenAI.
    elif user_plan == 'Monthly Plan':
        tier_1 = "openai_whisper"
        tier_2 = "assemblyai"
        tier_3 = "deepgram"
        reason = "monthly_plan_prioritizing_openai"

    # Free trial. Best model, so the first transcript someone ever sees
    # is the most accurate one we can produce.
    elif user_plan == 'free':
        tier_1 = "assemblyai"
        tier_2 = "openai_whisper"
        tier_3 = "deepgram"
        reason = "free_trial_prioritizing_assemblyai"

    # Safety net: any plan name that does not match one of the branches
    # above (a new plan added on the frontend, a naming mismatch, a
    # credit-only account with no active plan label, etc.) used to leave
    # every tier as None, which meant zero services were attempted and the
    # job failed instantly for a paying client. Never let an unrecognised
    # plan name produce an empty service list again.
    else:
        tier_1 = "openai_whisper"
        tier_2 = "assemblyai"
        tier_3 = "deepgram"
        reason = f"unmatched_plan_fallback_openai ('{user_plan}')"

    # --- Dynamic adjustment based on service availability ---
    final_tiers_list = []
    
    # Tier 1
    if tier_1 == "assemblyai" and ASSEMBLYAI_API_KEY:
        final_tiers_list.append("assemblyai")
    elif tier_1 == "openai_whisper" and OPENAI_WHISPER_SERVICE_RAILWAY_URL:
        final_tiers_list.append("openai_whisper")
    elif tier_1 == "deepgram" and DEEPGRAM_SERVICE_RAILWAY_URL:
        final_tiers_list.append("deepgram")

    # Tier 2 (only if not already in Tier 1 and is available)
    if tier_2 == "assemblyai" and ASSEMBLYAI_API_KEY and "assemblyai" not in final_tiers_list:
        final_tiers_list.append("assemblyai")
    elif tier_2 == "openai_whisper" and OPENAI_WHISPER_SERVICE_RAILWAY_URL and "openai_whisper" not in final_tiers_list:
        final_tiers_list.append("openai_whisper")
    elif tier_2 == "deepgram" and DEEPGRAM_SERVICE_RAILWAY_URL and "deepgram" not in final_tiers_list:
        final_tiers_list.append("deepgram")

    # Tier 3 (only if not already in Tier 1 or 2 and is available)
    if tier_3 == "assemblyai" and ASSEMBLYAI_API_KEY and "assemblyai" not in final_tiers_list:
        final_tiers_list.append("assemblyai")
    elif tier_3 == "openai_whisper" and OPENAI_WHISPER_SERVICE_RAILWAY_URL and "openai_whisper" not in final_tiers_list:
        final_tiers_list.append("openai_whisper")
    elif tier_3 == "deepgram" and DEEPGRAM_SERVICE_RAILWAY_URL and "deepgram" not in final_tiers_list:
        final_tiers_list.append("deepgram")

    # Ensure the list does not exceed 3 tiers and fills Nones if less than 3
    return {
        "tier_1": final_tiers_list[0] if len(final_tiers_list) > 0 else None,
        "tier_2": final_tiers_list[1] if len(final_tiers_list) > 1 else None,
        "tier_3": final_tiers_list[2] if len(final_tiers_list) > 2 else None,
        "reason": reason
    }


class PaystackVerificationRequest(BaseModel):
    reference: str

class PaystackInitializationRequest(BaseModel):
    email: str
    amount: float
    plan_name: str
    user_id: str
    country_code: str
    callback_url: str
    update_admin_revenue: Optional[bool] = False

class PaystackWebhookRequest(BaseModel):
    event: str
    data: dict

class TraineeRegistrationRequest(BaseModel):
    official_name: str
    country_code: str

class KoraVerificationRequest(BaseModel):
    reference: str

class CreditUpdateRequest(BaseModel):
    email: str
    plan_name: str
    amount: float
    currency: str
    duration_hours: Optional[int] = None
    duration_days: Optional[int] = None


class AdminCreditAdjustmentRequest(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    amount: int
    reason: str
    context: dict = {}

class FormattedWordDownloadRequest(BaseModel):
    transcription_html: str
    filename: Optional[str] = "transcription.docx"

class UserAIRequest_Pydantic(BaseModel):
    transcript: str
    user_prompt: str
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1000

class AdminAIFormatRequest_Pydantic(BaseModel):
    transcript: str
    formatting_instructions: str = "Format the transcript for readability, correct grammar, and identify main sections with headings. Ensure a professional tone."
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 4000

# Pydantic model for Gemini User queries (now available for all paid users)
class UserAIGeminiRequest_Pydantic(BaseModel):
    transcript: str
    user_prompt: str
    model: str = "models/gemini-pro-latest"
    max_tokens: int = 1000

# Pydantic model for Gemini Admin formatting
class AdminAIFormatGeminiRequest_Pydantic(BaseModel):
    transcript: str
    formatting_instructions: str = "Correct all grammar, ensure a formal tone, break into paragraphs with subheadings for each major topic, and highlight action items in bold."
    model: str = "models/gemini-pro-latest"
    max_tokens: int = 4000

jobs = {}
active_background_tasks = {}
cancellation_flags = {}

logger.info("Enhanced job tracking initialized")

# Firebase Firestore interaction functions
async def update_user_plan_firestore(user_id: str, new_plan: str, reference_id: Optional[str] = None, payment_amount_usd: Optional[float] = None):
    """Updates a user's plan and related fields in Firestore using Firebase Admin SDK."""
    if not db:
        logger.error("Firestore client not initialized. Cannot update user plan.")
        return {'success': False, 'error': 'Firestore not initialized'}

    user_ref = db.collection('users').document(user_id)
    if reference_id:
        try:
            existing_snapshot = await asyncio.to_thread(user_ref.get)
            existing_data = existing_snapshot.to_dict() if existing_snapshot.exists else {}
            if existing_data.get('paystackReferenceId') == reference_id or existing_data.get('paymentReferenceId') == reference_id:
                logger.info(f"Payment {reference_id} already applied for user {user_id}; ignoring duplicate.")
                return {'success': True, 'already_applied': True}
        except Exception as e:
            logger.warning(f"Could not check duplicate payment reference {reference_id}: {e}")
    updates = {
        'plan': new_plan,
        'lastAccessed': firestore.SERVER_TIMESTAMP,
        'paystackReferenceId': reference_id,
        'paymentReferenceId': reference_id,
        'hasReceivedInitialFreeMinutes': True,
        'totalMinutesUsed': 0
    }

    # Buying a plan refills the plan purse. Bought top-up credits are left
    # alone deliberately: a client who paid for them keeps them.
    updates.update(grant_plan_credits(new_plan))

    plan_duration_days = 0
    if new_plan == 'One-Day Plan':
        plan_duration_days = 1
    elif new_plan == 'Three-Day Plan':
        plan_duration_days = 3
    elif new_plan == 'One-Week Plan':
        plan_duration_days = 7
    elif new_plan == 'Monthly Plan':
        plan_duration_days = 30
    elif new_plan == 'Yearly Plan':
        plan_duration_days = 365

    if plan_duration_days > 0:
        expires_at = datetime.now() + timedelta(days=plan_duration_days)
        updates['expiresAt'] = expires_at
        updates['subscriptionStartDate'] = firestore.SERVER_TIMESTAMP
        logger.info(f"User {user_id} {new_plan} plan will expire on: {expires_at}")
    else:
        updates['expiresAt'] = None
        updates['subscriptionStartDate'] = None

    try:
        await asyncio.to_thread(user_ref.update, updates)
        granted = _int(updates.get("planCredits"))
        if granted:
            await _record_credit_ledger(
                user_id, granted, "plan purchase",
                {"plan": new_plan, "payment_reference": reference_id, "payment_amount_usd": payment_amount_usd},
                granted + _int((await asyncio.to_thread(user_ref.get)).to_dict().get("topUpCredits")),
            )
        logger.info(f"User {user_id} plan updated to {new_plan} in Firestore.")
        return {'success': True}
    except Exception as e:
        logger.error(f"Error updating user {user_id} plan in Firestore: {e}")
        return {'success': False, 'error': str(e)}

async def update_monthly_revenue_firestore(amount: float):
    """Updates the cumulative monthly revenue in Firestore."""
    if not db:
        logger.error("Firestore client not initialized. Cannot update monthly revenue.")
        return {'success': False, 'error': 'Firestore not initialized'}

    admin_stats_ref = db.collection('admin_stats').document('current')
    try:
        await asyncio.to_thread(admin_stats_ref.update, {
            'monthlyRevenue': firestore.Increment(amount)
        })
        logger.info(f"Monthly revenue updated by {amount} in Firestore.")
        return {'success': True}
    except Exception as e:
        logger.error(f"Error updating monthly revenue in Firestore: {e}")
        return {'success': False, 'error': str(e)}

async def get_user_profile_by_email_firestore(email: str):
    """Fetches user profile by email to get UID (for webhook processing)."""
    if not db:
        logger.error("Firestore client not initialized. Cannot fetch user by email.")
        return None
    try:
        users_ref = db.collection('users')
        query_ref = users_ref.where(filter=FieldFilter("email", "==", email)).limit(1)
        snapshot = await asyncio.to_thread(query_ref.get) 

        for doc in snapshot:
            return doc.id
        return None
    except Exception as e:
        logger.error(f"Error fetching user by email {email}: {e}")
        return None

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
        
        # 44,100 Hz is deliberate, not wasteful. Below 32,000 Hz the MP3
        # encoder switches to the MPEG-2 and MPEG-2.5 variants, and MPEG-2.5
        # in particular is a non-standard extension that professional
        # playback software such as ExpressScribe refuses to open. Clients
        # were having to convert every recording by hand. Mono at 64 kbps
        # keeps the file small; the sample rate is what buys compatibility.
        target_sample_rate = PLAYABLE_SAMPLE_RATE
        audio = audio.set_frame_rate(target_sample_rate)
        logger.info(f"Set sample rate to {target_sample_rate} Hz for playback compatibility")
        
        if job_id and cancellation_flags.get(job_id, False):
            logger.info(f"Job {job_id} cancelled during sample rate conversion")
            raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        audio = audio - 3
        audio = audio.normalize()
        logger.info("Applied audio normalization")
        
        if job_id and cancellation_flags.get(job_id, False):
            logger.info(f"Job {job_id} cancelled before export")
            raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        # No -q:a here. Passing a quality setting alongside a bitrate makes
        # the encoder ignore the bitrate entirely: "-q:a 9" was turning a
        # requested 64 kbps into roughly 10 kbps, which is why recordings
        # came back sounding poor.
        audio.export(
            output_path,
            format="mp3",
            bitrate=PLAYABLE_BITRATE,
            parameters=[
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
            audio = audio.set_frame_rate(PLAYABLE_SAMPLE_RATE)
            audio.export(output_path, format="mp3", bitrate=PLAYABLE_BITRATE)
            
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
        
        # Every quality now stays at 44,100 Hz. Only the bitrate changes.
        # Dropping the sample rate saves very little and costs compatibility,
        # because anything under 32,000 Hz leaves standard MPEG-1 audio.
        # "voice" is what the app's own recordings use. They are captured as
        # 32 kbps mono Opus, so re-encoding them to 128 kbps stereo made the
        # file roughly four times larger than the original while adding no
        # quality at all -- you cannot put back detail that was never
        # recorded. 48 kbps mono at 44,100 Hz is transparent against that
        # source, stays standard MPEG-1 so playback software still opens it,
        # and is about two and a half times smaller than the old setting.
        if quality == "voice":
            bitrate = "48k"
            sample_rate = PLAYABLE_SAMPLE_RATE
            channels = 1
        elif quality == "high":
            bitrate = "128k"
            sample_rate = PLAYABLE_SAMPLE_RATE
            channels = 2 if audio.channels > 1 else 1
        elif quality == "medium":
            bitrate = "96k"
            sample_rate = PLAYABLE_SAMPLE_RATE
            channels = 1
        else:
            bitrate = PLAYABLE_BITRATE
            sample_rate = PLAYABLE_SAMPLE_RATE
            channels = 1
        
        if audio.channels != channels:
            audio = audio.set_channels(channels)
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)
        
        # Again no -q:a, so the chosen bitrate is the bitrate we actually get.
        audio.export(
            output_path,
            format="mp3",
            bitrate=bitrate,
            parameters=[
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
    'OTHER_AFRICA': 'USD', # 'USD' indicates no conversion, direct USD payment
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

def get_local_amount_and_currency(base_usd_amount: float, country_code: str, plan_name: str = None) -> tuple[float, str]:
    # REMOVED: 'Monthly Plan' from this condition. Now only Yearly Plan forces USD.
    if plan_name in ['Yearly Plan']:
        return base_usd_amount, 'USD'

    currency = COUNTRY_CURRENCY_MAP.get(country_code, 'USD')
    if currency == 'USD':
        return base_usd_amount, 'USD'
    
    rate = USD_TO_LOCAL_RATES.get(country_code, 1.0)
    local_amount = round(base_usd_amount * rate, 2)
    return local_amount, currency

def get_payment_channels(country_code: str, plan_name: str = None) -> list[str]:
    # REMOVED: 'Monthly Plan' from this condition. Now only Yearly Plan forces card payments.
    if plan_name in ['Yearly Plan']:
        return ['card']
    return COUNTRY_CHANNELS_MAP.get(country_code, ['card'])

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
            logger.info(f"Paystack verification raw response for {reference}: {payment_data}") # Added for debugging

            # Safely check if 'status' is true and 'data' is a dictionary with 'status' as 'success'
            if payment_data.get('status') is True and \
               isinstance(payment_data.get('data'), dict) and \
               payment_data['data'].get('status') == 'success':

                amount_kobo = payment_data['data']['amount']
                amount = amount_kobo / 100
                customer_email = payment_data['data']['customer']['email']
                # Corrected: Access currency directly from payment_data['data']
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
                logger.warning(f"❌ Paystack payment verification failed (non-success status or missing 'data' key): {payment_data}")
                return {
                    'status': 'failed',
                    'error': payment_data.get('message', 'Payment verification failed'),
                    'raw_data': payment_data
                }
        else:
            logger.error(f"❌ Paystack API error (non-200 status): {response.status_code} - {response.text}")
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
        # Log the type of exception for better debugging
        logger.error(f"❌ Unexpected error during Paystack verification: {type(e).__name__}: {str(e)}")
        return {
            'status': 'error',
            "error": 'Payment verification failed',
            'details': str(e)
        }

async def enroll_paid_trainee(email: str, reference: str, amount: float, currency: str, country_code: str, payment_provider: str = "paystack", user_id: Optional[str] = None):
    """Unlock Training Room only after the payment provider confirms success.

    When the account was just created, the Firebase auth event and the profile
    document can arrive at different times.  A signed-in finalization request
    supplies the verified UID directly so enrollment never depends on a
    profile lookup racing the auth callback.
    """
    if not db:
        return {"success": False, "error": "Firestore not initialized"}
    user_id = user_id or await get_user_profile_by_email_firestore(email)
    if not user_id:
        return {"success": False, "error": f"User {email} not found in Firestore."}
    profile = await _load_profile(user_id) or {}
    if profile.get("lastTrainingPaymentReference") == reference:
        return {"success": True, "already_applied": True, "trainee_enrolled": True}
    if str(country_code or "").upper() != TRAINEE_COUNTRY:
        return {"success": False, "error": "Training enrollment is currently limited to Kenya."}
    updates = {
        "role": "trainee",
        "traineeStatus": "enrolled",
        "trainingStatus": "active",
        "trainingLevel": max(1, int(profile.get("trainingLevel") or 1)),
        "trainingPaymentStatus": "paid",
        "trainingPaymentReference": reference,
        "lastTrainingPaymentReference": reference,
        "trainingPaymentProvider": payment_provider,
        "trainingPaymentAmountUsd": TRAINEE_PRICE_USD,
        "trainingPaymentCurrency": currency,
        "trainingRoomAccess": True,
        "workerApproved": False,
        "traineeAccountPendingDeletion": False,
        "trainingPaidAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    await asyncio.to_thread(db.collection("users").document(user_id).set, updates, merge=True)
    logger.info("Paid trainee enrolled in Training Room: %s", email)
    return {"success": True, "trainee_enrolled": True, "email": email, "reference": reference}


async def update_user_credits_paystack(email: str, plan_name: str, amount: float, currency: str, update_admin_revenue: bool = False, country_code: Optional[str] = None, reference: Optional[str] = None):
    """
    Update user credits/plan in Firestore.
    The real-time revenue counter logic is now handled purely on the frontend.
    """
    if not db:
        logger.error(f"Firestore client not initialized. Cannot update credits for {email}.")
        return {'success': False, 'error': 'Firestore not initialized'}

    try:
        logger.info(f"📝 Updating credits for {email} - {plan_name} ({amount} {currency}) in Firestore.")
        
        # 1. Get user UID from email
        user_id = await get_user_profile_by_email_firestore(email)
        if not user_id:
            logger.error(f"User with email {email} not found in Firestore. Cannot update plan.")
            return {'success': False, 'error': f"User {email} not found in Firestore."}

        # 2a. Trainee enrollment is a product, not an AI plan or credit top-up.
        if plan_name == TRAINEE_PRODUCT:
            return await enroll_paid_trainee(email, reference or "", amount, currency, country_code or "", "paystack")

        # 2b. A top-up is not a plan. Buying credits adds them to the bought
        # purse and leaves the plan, the free-trial flag and everything else
        # exactly as it was. Both the callback and the webhook can arrive for
        # the same payment, so the reference is remembered and a repeat is
        # ignored rather than granting the credits twice.
        if plan_name in TOPUP_BUNDLES or custom_topup_credits(plan_name, country_code or 'KE') is not None:
            profile = await _load_profile(user_id) or {}
            if reference and profile.get('lastTopUpReference') == reference:
                logger.info(f"Top-up {reference} for {email} already applied; ignoring the repeat.")
                return {'success': True, 'email': email, 'plan': plan_name, 'amount': amount,
                        'currency': currency, 'already_applied': True}
            result = grant_topup_credits(profile, plan_name, country_code=country_code or 'KE')
            if not result:
                return {'success': False, 'error': 'Unknown top-up bundle'}
            topup_updates = dict(result['updates'])
            topup_updates['lastAccessed'] = firestore.SERVER_TIMESTAMP
            topup_updates['lastTopUpAt'] = firestore.SERVER_TIMESTAMP
            if reference:
                topup_updates['lastTopUpReference'] = reference
            await asyncio.to_thread(db.collection('users').document(user_id).update, topup_updates)
            await _record_credit_ledger(
                user_id, result['added'], "credit top-up purchase",
                {"item": plan_name, "payment_reference": reference, "amount": amount, "currency": currency},
                result['newTotal'],
            )
            if update_admin_revenue:
                await update_monthly_revenue_firestore(amount)
            logger.info(f"Added {result['added']} bought credits to {email}.")
            return {'success': True, 'email': email, 'plan': plan_name, 'amount': amount,
                    'currency': currency, 'credits_added': result['added']}

        # 2b. Set the plan AND fill the credit purse that comes with it.
        #
        # This used to write the plan by hand here and never grant a single
        # credit, while the routine that does grant them sat unused. The
        # result was a client who had paid, held a valid plan, and could not
        # transcribe a thing. One routine now does both, so the two can never
        # drift apart again.
        plan_result = await update_user_plan_firestore(user_id, plan_name, reference, amount)
        if not plan_result.get('success'):
            return {'success': False, 'error': plan_result.get('error', 'Could not update the plan.')}

        # 3. Update monthly revenue if flag is True
        if update_admin_revenue:
            revenue_update_result = await update_monthly_revenue_firestore(amount)
            if revenue_update_result['success']:
                logger.info(f"✅ Monthly revenue updated successfully for payment of {amount} {currency}.")
            else:
                logger.warning(f"⚠️ Failed to update monthly revenue: {revenue_update_result.get('error')}")
        
        logger.info(f"✅ Credits and plan updated successfully for {email} in Firestore.")
        return {'success': True, 'email': email, 'plan': plan_name, 'amount': amount, 'currency': currency}
        
    except Exception as e:
        logger.error(f"❌ Error updating user credits in Firestore: {str(e)}")
        return {'success': False, 'error': str(e)}

async def transcribe_with_openai_whisper(audio_path: str, language_code: str, job_id: str) -> dict:
    """Calls the dedicated OpenAI Whisper service deployed on Render."""
    if not OPENAI_WHISPER_SERVICE_RAILWAY_URL:
        logger.error(f"{TYPEMYWORDZ2_NAME} Service URL not configured, skipping {TYPEMYWORDZ2_NAME} for job {job_id}")
        return {
            "status": "failed",
            "error": "Transcription service unavailable. Please try again later."
        }

    try:
        logger.info(f"Calling {TYPEMYWORDZ2_NAME} service for job {job_id} at {OPENAI_WHISPER_SERVICE_RAILWAY_URL}/transcribe")
        
        # Read the audio file content
        with open(audio_path, "rb") as f:
            audio_content = f.read()

        # Prepare form data
        files = {'file': (os.path.basename(audio_path), audio_content, 'audio/mpeg')}
        data = {'language_code': language_code}

        # Make HTTP POST request to the dedicated Whisper service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_WHISPER_SERVICE_RAILWAY_URL}/transcribe",
                files=files,
                data=data,
                timeout=300.0
            )
            response.raise_for_status()

        result = response.json()
        
        if result.get("status") == "completed" and result.get("transcription"):
            logger.info(f"{TYPEMYWORDZ2_NAME} service transcription completed for job {job_id}")
            return result
        else:
            raise Exception(f"{TYPEMYWORDZ2_NAME} service returned an incomplete or failed status: {result}")

    except asyncio.CancelledError:
        logger.info(f"{TYPEMYWORDZ2_NAME} service call cancelled for job {job_id}")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"{TYPEMYWORDZ2_NAME} service HTTP error for job {job_id}: {e.response.status_code} - {e.response.text}")
        return {
            "status": "failed",
            "error": "Transcription failed due to a service error. Please try again."
        }
    except httpx.RequestError as e:
        logger.error(f"{TYPEMYWORDZ2_NAME} service network error for job {job_id}: {e}")
        return {
            "status": "failed",
            "error": "Transcription failed due to a network issue. Please check your connection and try again."
        }
    except Exception as e:
        logger.error(f"{TYPEMYWORDZ2_NAME} transcription failed for job {job_id}: {str(e)}")
        return {
            "status": "failed",
            "error": "Transcription failed. Please try again later."
        }
    finally:
        pass

def build_segments_from_words(words: list, max_chars: int = 240) -> list:
    """Group word-level timings into sentence-sized, clickable segments.

    A transcript without speaker labels arrives as one long block of prose.
    Handing the editor every individual word would be unusable, so words are
    gathered up until a sentence ends or the line gets too long. Each segment
    keeps its own start and end time and the lowest word confidence it
    contains, which is what lets the editor flag words worth checking.
    """
    segments = []
    current = []

    def flush():
        if not current:
            return
        text = " ".join(w.get("text", "") for w in current).strip()
        if not text:
            current.clear()
            return
        confidences = [w.get("confidence") for w in current if w.get("confidence") is not None]
        segments.append({
            "start": round(current[0].get("start", 0) / 1000.0, 3),
            "end": round(current[-1].get("end", 0) / 1000.0, 3),
            "speaker": None,
            "text": text,
            "confidence": round(min(confidences), 4) if confidences else None
        })
        current.clear()

    for word in words:
        current.append(word)
        text = (word.get("text") or "").strip()
        ends_sentence = text.endswith((".", "?", "!"))
        too_long = sum(len(w.get("text", "")) + 1 for w in current) >= max_chars
        if ends_sentence or too_long:
            flush()

    flush()
    return segments

async def transcribe_with_assemblyai(audio_path: str, language_code: str, speaker_labels_enabled: bool, model: str, job_id: str) -> dict:
    """Transcribe audio using AssemblyAI API"""
    if not ASSEMBLYAI_API_KEY:
        logger.error(f"{TYPEMYWORDZ1_NAME} API Key not configured, skipping {TYPEMYWORDZ1_NAME} for job {job_id}")
        return {
            "status": "failed",
            "error": "Transcription service unavailable. Please try again later."
        }

    try:
        logger.info(f"Starting {TYPEMYWORDZ1_NAME} transcription with {model} model for job {job_id}")
        
        def check_cancellation():
            if job_id and cancellation_flags.get(job_id, False):
                logger.info(f"Job {job_id} was cancelled during {TYPEMYWORDZ1_NAME} processing")
                raise asyncio.CancelledError(f"Job {job_id} was cancelled")
        
        check_cancellation()
        
        compressed_path, compression_stats = compress_audio_for_transcription(audio_path, job_id=job_id)
        logger.info(f"Audio compressed for {TYPEMYWORDZ1_NAME}: {compression_stats}")

        check_cancellation()
        
        logger.info(f"Uploading audio to {TYPEMYWORDZ1_NAME}...")
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        upload_endpoint = "https://api.assemblyai.com/v2/upload"
        
        with open(compressed_path, "rb") as f:
            upload_response = requests.post(upload_endpoint, headers=headers, data=f)
        
        if upload_response.status_code != 200:
            raise Exception(f"Failed to upload audio to {TYPEMYWORDZ1_NAME}: {upload_response.text}")
        
        upload_result = upload_response.json()
        audio_url = upload_result["upload_url"]
        logger.info(f"Audio uploaded to {TYPEMYWORDZ1_NAME}: {audio_url}")

        check_cancellation()

        headers = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}
        transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
        json_data = {
            "audio_url": audio_url,
            "language_code": language_code,
            "punctuate": True,
            "format_text": True,
            "speaker_labels": speaker_labels_enabled,
            "speech_models": model,
        }
        
        transcript_response = requests.post(transcript_endpoint, headers=headers, json=json_data)
        
        if transcript_response.status_code != 200:
            raise Exception(f"Failed to start transcription on {TYPEMYWORDZ1_NAME}: {transcript_response.text}")
        
        transcript_result = transcript_response.json()
        transcript_id = transcript_result["id"]
        logger.info(f"{TYPEMYWORDZ1_NAME} transcription started with ID: {transcript_id}")
        
        if os.path.exists(compressed_path):
            os.unlink(compressed_path)
            logger.info(f"Cleaned up compressed file: {compressed_path}")

        while True:
            check_cancellation()
            
            await asyncio.sleep(5)
            
            status_response = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers={"authorization": ASSEMBLYAI_API_KEY})
            
            if status_response.status_code != 200:
                raise Exception(f"Failed to get status from {TYPEMYWORDZ1_NAME}: {status_response.text}")
            
            status_result = status_response.json()
            
            if status_result["status"] == "completed":
                transcription_text = status_result["text"]
                
                segments = []

                if speaker_labels_enabled and status_result.get("utterances"):
                    formatted_transcript = ""
                    for utterance in status_result.get("utterances"):
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
                            speaker_num = str(ord(speaker_letter.upper()) - ord('A') + 1)
                        
                        formatted_transcript += f"<strong>Speaker {speaker_num}:</strong> {utterance['text']}\n"

                        # One segment per utterance, in the same order as the
                        # lines above, so the editor can pair line N of the
                        # transcript with segment N without re-parsing text.
                        segments.append({
                            "start": round(utterance.get("start", 0) / 1000.0, 3),
                            "end": round(utterance.get("end", 0) / 1000.0, 3),
                            "speaker": f"Speaker {speaker_num}",
                            "text": utterance.get("text", ""),
                            "confidence": utterance.get("confidence")
                        })
                    transcription_text = formatted_transcript.strip()
                else:
                    # No speaker labels, so the transcript is one block of prose.
                    # Group the word-level timings into sentences to give the
                    # editor something of a sensible size to jump between.
                    segments = build_segments_from_words(status_result.get("words") or [])

                return {
                    "status": "completed",
                    "transcription": transcription_text,
                    "language": status_result["language_code"],
                    "duration": status_result.get("audio_duration", 0),
                    "word_count": len(transcription_text.split()) if transcription_text else 0,
                    "has_speaker_labels": speaker_labels_enabled and bool(status_result.get("utterances")),
                    "segments": segments,
                    "timings_source": TYPEMYWORDZ1_NAME,
                    # Which model actually ran. We send a preference list,
                    # so this can differ from what was requested.
                    "model_used": status_result.get("speech_model_used")
                }
            elif status_result["status"] == "error":
                raise Exception(status_result.get("error", f"Transcription failed on {TYPEMYWORDZ1_NAME}"))
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
            "error": "Transcription failed. Please try again later."
        }

async def transcribe_with_deepgram(audio_path: str, language_code: str, speaker_labels_enabled: bool, job_id: str) -> dict:
    """Calls the dedicated Deepgram service deployed on Render."""
    if not DEEPGRAM_SERVICE_RAILWAY_URL:
        logger.error(f"{DEEPGRAM_NAME} Service URL not configured, skipping {DEEPGRAM_NAME} for job {job_id}")
        return {
            "status": "failed",
            "error": "Transcription service unavailable. Please try again later."
        }

    try:
        logger.info(f"Calling {DEEPGRAM_NAME} service for job {job_id} at {DEEPGRAM_SERVICE_RAILWAY_URL}/transcribe")
        
        # Read the audio file content
        with open(audio_path, "rb") as f:
            audio_content = f.read()

        # Prepare form data
        files = {'file': (os.path.basename(audio_path), audio_content, 'audio/mpeg')}
        data = {
            'language_code': language_code,
            'speaker_labels_enabled': str(speaker_labels_enabled).lower()
        }

        # Make HTTP POST request to the dedicated Deepgram service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DEEPGRAM_SERVICE_RAILWAY_URL}/transcribe",
                files=files,
                data=data,
                timeout=300.0
            )
            response.raise_for_status()

        result = response.json()
        
        if result.get("status") == "completed" and result.get("transcription"):
            logger.info(f"{DEEPGRAM_NAME} service transcription completed for job {job_id}")
            return result
        else:
            raise Exception(f"{DEEPGRAM_NAME} service returned an incomplete or failed status: {result}")

    except asyncio.CancelledError:
        logger.info(f"{DEEPGRAM_NAME} service call cancelled for job {job_id}")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"{DEEPGRAM_NAME} service HTTP error for job {job_id}: {e.response.status_code} - {e.response.text}")
        return {
            "status": "failed",
            "error": "Transcription failed due to a service error. Please try again."
        }
    except httpx.RequestError as e:
        logger.error(f"{DEEPGRAM_NAME} service network error for job {job_id}: {e}")
        return {
            "status": "failed",
            "error": "Transcription failed due to a network issue. Please check your connection and try again."
        }
    except Exception as e:
        logger.error(f"{DEEPGRAM_NAME} transcription failed for job {job_id}: {str(e)}")
        return {
            "status": "failed",
            "error": "Transcription failed. Please try again later."
        }
    finally:
        pass

async def process_transcription_job(job_id: str, tmp_path: str, filename: str, language_code: Optional[str], speaker_labels_enabled: bool, user_plan: str, duration_minutes: float, user_email: str = ""):
    """Updated transcription processing with new three-tier service logic and admin/tester email checking."""
    logger.info(f"Starting transcription job {job_id}: {filename}, duration: {duration_minutes:.1f}min, plan: {user_plan}, email: {user_email}, speaker_labels: {speaker_labels_enabled}")
    job_data = jobs[job_id]
    
    active_background_tasks[job_id] = asyncio.current_task()
    cancellation_flags[job_id] = False

    compressed_path = None

    try:
        def check_cancellation():
            if cancellation_flags.get(job_id, False) or job_data.get("status") == "cancelled":
                logger.info(f"Job {job_id} was cancelled - stopping processing")
                raise asyncio.CancelledError(f"Job {job_id} was cancelled")
            return True

        check_cancellation()

        # Get service configuration based on new logic
        service_config = get_transcription_services(user_plan, speaker_labels_enabled, user_email)
        tier_1_service = service_config["tier_1"]
        tier_2_service = service_config["tier_2"]
        tier_3_service = service_config["tier_3"]

        logger.info(f"🎯 Job {job_id} service selection: Tier1={tier_1_service}, Tier2={tier_2_service}, Tier3={tier_3_service} ({service_config['reason']})")

        def get_assemblyai_model(plan: str) -> list:
            # Everyone, including free-trial users, gets the highest
            # accuracy model. universal-2 is listed second so that
            # languages universal-3-5-pro does not support still work
            # instead of failing outright.
            return ["universal-3-5-pro", "universal-2"]

        assemblyai_model = get_assemblyai_model(user_plan)

        job_data.update({
            "tier_1_service": tier_1_service,
            "tier_2_service": tier_2_service,
            "tier_3_service": tier_3_service,
            "assemblyai_model": assemblyai_model,
            "duration_minutes": duration_minutes,
            "selection_reason": service_config["reason"],
            "user_email": user_email,
            "is_admin": is_admin_user(user_email)
        })

        transcription_result = None
        services_attempted = []

        # --- ATTEMPT TIER 1 SERVICE ---
        if tier_1_service == "assemblyai":
            if not ASSEMBLYAI_API_KEY:
                logger.error(f"{TYPEMYWORDZ1_NAME} API Key not configured, skipping Tier 1 for job {job_id}")
                job_data["tier_1_error"] = f"{TYPEMYWORDZ1_NAME} API Key not configured"
            else:
                try:
                    logger.info(f"🚀 Attempting {TYPEMYWORDZ1_NAME} (Tier 1 Primary) for job {job_id}")
                    compressed_path, compression_stats = compress_audio_for_transcription(tmp_path, job_id=job_id)
                    logger.info(f"Audio compressed for {TYPEMYWORDZ1_NAME}: {compression_stats}")
                    transcription_result = await transcribe_with_assemblyai(compressed_path, language_code, speaker_labels_enabled, assemblyai_model, job_id)
                    job_data["tier_1_used"] = "assemblyai"
                    job_data["tier_1_success"] = True
                except Exception as error:
                    logger.error(f"❌ {TYPEMYWORDZ1_NAME} (Tier 1 Primary) failed: {error}")
                    job_data["tier_1_error"] = str(error)
                    job_data["tier_1_success"] = False
            services_attempted.append(f"{TYPEMYWORDZ1_NAME}_tier1")
            
        elif tier_1_service == "openai_whisper":
            if not OPENAI_WHISPER_SERVICE_RAILWAY_URL:
                logger.error(f"{TYPEMYWORDZ2_NAME} Service URL not configured, skipping Tier 1 for job {job_id}")
                job_data["tier_1_error"] = f"{TYPEMYWORDZ2_NAME} Service URL not configured"
            else:
                try:
                    logger.info(f"🚀 Attempting {TYPEMYWORDZ2_NAME} (Tier 1 Primary) for job {job_id}")
                    compressed_path, compression_stats = compress_audio_for_transcription(tmp_path, job_id=job_id)
                    logger.info(f"Audio compressed for {TYPEMYWORDZ2_NAME}: {compression_stats}")
                    transcription_result = await transcribe_with_openai_whisper(compressed_path, language_code, job_id)
                    job_data["tier_1_used"] = "openai_whisper"
                    job_data["tier_1_success"] = True
                except Exception as error:
                    logger.error(f"❌ {TYPEMYWORDZ2_NAME} (Tier 1 Primary) failed: {error}")
                    job_data["tier_1_error"] = str(error)
                    job_data["tier_1_success"] = False
            services_attempted.append(f"{TYPEMYWORDZ2_NAME}_tier1")
        
        elif tier_1_service is not None and tier_1_service not in ("assemblyai", "openai_whisper", "deepgram"):
            # Belt and braces. If the router ever names a service the runner
            # does not know, say so loudly instead of falling through every
            # branch and reporting a failure that looks like bad audio.
            logger.error(f"Unknown tier 1 service '{tier_1_service}' for job {job_id}")
            job_data["tier_1_error"] = f"Unknown service '{tier_1_service}'"

        elif tier_1_service == "deepgram":
            if not DEEPGRAM_SERVICE_RAILWAY_URL:
                logger.error(f"{DEEPGRAM_NAME} Service URL not configured, skipping Tier 1 for job {job_id}")
                job_data["tier_1_error"] = f"{DEEPGRAM_NAME} Service URL not configured"
            else:
                try:
                    logger.info(f"🚀 Attempting {DEEPGRAM_NAME} (Tier 1 Primary) for job {job_id}")
                    compressed_path, compression_stats = compress_audio_for_transcription(tmp_path, job_id=job_id)
                    logger.info(f"Audio compressed for {DEEPGRAM_NAME}: {compression_stats}")
                    transcription_result = await transcribe_with_deepgram(compressed_path, language_code, speaker_labels_enabled, job_id)
                    job_data["tier_1_used"] = "deepgram"
                    job_data["tier_1_success"] = True
                except Exception as error:
                    logger.error(f"❌ {DEEPGRAM_NAME} (Tier 1 Primary) failed: {error}")
                    job_data["tier_1_error"] = str(error)
                    job_data["tier_1_success"] = False
            services_attempted.append(f"{DEEPGRAM_NAME}_tier1")
        
        check_cancellation()

        # --- ATTEMPT TIER 2 SERVICE (FALLBACK 1) if Tier 1 failed AND tier_2_service is defined ---
        if (not transcription_result or transcription_result.get("status") == "failed") and tier_2_service:
            logger.warning(f"⚠️ Tier 1 service failed, trying Tier 2 fallback ({tier_2_service}) for job {job_id}")
            
            if tier_2_service == "assemblyai":
                if not ASSEMBLYAI_API_KEY:
                    logger.error(f"{TYPEMYWORDZ1_NAME} API Key not configured, skipping Tier 2 for job {job_id}")
                    job_data["tier_2_error"] = f"{TYPEMYWORDZ1_NAME} API Key not configured"
                else:
                    try:
                        logger.info(f"🔄 Attempting {TYPEMYWORDZ1_NAME} (Tier 2 Fallback) for job {job_id}")
                        if compressed_path is None:
                            compressed_path, compression_stats = compress_audio_for_transcription(tmp_path, job_id=job_id)
                            logger.info(f"Audio compressed for {TYPEMYWORDZ1_NAME}: {compression_stats}")
                        transcription_result = await transcribe_with_assemblyai(compressed_path, language_code, speaker_labels_enabled, assemblyai_model, job_id)
                        job_data["tier_2_used"] = "assemblyai"
                        job_data["tier_2_success"] = True
                    except Exception as error:
                        logger.error(f"❌ {TYPEMYWORDZ1_NAME} (Tier 2 Fallback) failed: {error}")
                        job_data["tier_2_error"] = str(error)
                        job_data["tier_2_success"] = False
                services_attempted.append(f"{TYPEMYWORDZ1_NAME}_tier2")
                
            elif tier_2_service == "openai_whisper":
                if not OPENAI_WHISPER_SERVICE_RAILWAY_URL:
                    logger.error(f"{TYPEMYWORDZ2_NAME} Service URL not configured, skipping Tier 2 for job {job_id}")
                    job_data["tier_2_error"] = f"{TYPEMYWORDZ2_NAME} Service URL not configured"
                else:
                    try:
                        logger.info(f"🔄 Attempting {TYPEMYWORDZ2_NAME} (Tier 2 Fallback) for job {job_id}")
                        if compressed_path is None:
                            compressed_path, compression_stats = compress_audio_for_transcription(tmp_path, job_id=job_id)
                            logger.info(f"Audio compressed for {TYPEMYWORDZ2_NAME}: {compression_stats}")
                        transcription_result = await transcribe_with_openai_whisper(compressed_path, language_code, job_id)
                        job_data["tier_2_used"] = "openai_whisper"
                        job_data["tier_2_success"] = True
                    except Exception as error:
                        logger.error(f"❌ {TYPEMYWORDZ2_NAME} (Tier 2 Fallback) failed: {error}")
                        job_data["tier_2_error"] = str(error)
                        job_data["tier_2_success"] = False
                services_attempted.append(f"{TYPEMYWORDZ2_NAME}_tier2")
            
            elif tier_2_service == "deepgram":
                if not DEEPGRAM_SERVICE_RAILWAY_URL:
                    logger.error(f"{DEEPGRAM_NAME} Service URL not configured, skipping Tier 2 for job {job_id}")
                    job_data["tier_2_error"] = f"{DEEPGRAM_NAME} Service URL not configured"
                else:
                    try:
                        logger.info(f"🔄 Attempting {DEEPGRAM_NAME} (Tier 2 Fallback) for job {job_id}")
                        if compressed_path is None:
                            compressed_path, compression_stats = compress_audio_for_transcription(tmp_path, job_id=job_id)
                            logger.info(f"Audio compressed for {DEEPGRAM_NAME}: {compression_stats}")
                        transcription_result = await transcribe_with_deepgram(compressed_path, language_code, speaker_labels_enabled, job_id)
                        job_data["tier_2_used"] = "deepgram"
                        job_data["tier_2_success"] = True
                    except Exception as error:
                        logger.error(f"❌ {DEEPGRAM_NAME} (Tier 2 Fallback) failed: {error}")
                        job_data["tier_2_error"] = str(error)
                        job_data["tier_2_success"] = False
                services_attempted.append(f"{DEEPGRAM_NAME}_tier2")
            
        check_cancellation()

        # --- ATTEMPT TIER 3 SERVICE (FALLBACK 2) if Tier 2 failed AND tier_3_service is defined ---
        if (not transcription_result or transcription_result.get("status") == "failed") and tier_3_service:
            logger.warning(f"⚠️ Tier 2 service failed, trying Tier 3 fallback ({tier_3_service}) for job {job_id}")
            
            if tier_3_service == "assemblyai":
                if not ASSEMBLYAI_API_KEY:
                    logger.error(f"{TYPEMYWORDZ1_NAME} API Key not configured, skipping Tier 3 for job {job_id}")
                    job_data["tier_3_error"] = f"{TYPEMYWORDZ1_NAME} API Key not configured"
                else:
                    try:
                        logger.info(f"🔄 Attempting {TYPEMYWORDZ1_NAME} (Tier 3 Fallback) for job {job_id}")
                        if compressed_path is None:
                            compressed_path, compression_stats = compress_audio_for_transcription(tmp_path, job_id=job_id)
                            logger.info(f"Audio compressed for {TYPEMYWORDZ1_NAME}: {compression_stats}")
                        transcription_result = await transcribe_with_assemblyai(compressed_path, language_code, speaker_labels_enabled, assemblyai_model, job_id)
                        job_data["tier_3_used"] = "assemblyai"
                        job_data["tier_3_success"] = True
                    except Exception as error:
                        logger.error(f"❌ {TYPEMYWORDZ1_NAME} (Tier 3 Fallback) failed: {error}")
                        job_data["tier_3_error"] = str(error)
                        job_data["tier_3_success"] = False
                services_attempted.append(f"{TYPEMYWORDZ1_NAME}_tier3")
                
            elif tier_3_service == "openai_whisper":
                if not OPENAI_WHISPER_SERVICE_RAILWAY_URL:
                    logger.error(f"{TYPEMYWORDZ2_NAME} Service URL not configured, skipping Tier 3 for job {job_id}")
                    job_data["tier_3_error"] = f"{TYPEMYWORDZ2_NAME} Service URL not configured"
                else:
                    try:
                        logger.info(f"🔄 Attempting {TYPEMYWORDZ2_NAME} (Tier 3 Fallback) for job {job_id}")
                        if compressed_path is None:
                            compressed_path, compression_stats = compress_audio_for_transcription(tmp_path, job_id=job_id)
                            logger.info(f"Audio compressed for {TYPEMYWORDZ2_NAME}: {compression_stats}")
                        transcription_result = await transcribe_with_openai_whisper(compressed_path, language_code, job_id)
                        job_data["tier_3_used"] = "openai_whisper"
                        job_data["tier_3_success"] = True
                    except Exception as error:
                        logger.error(f"❌ {TYPEMYWORDZ2_NAME} (Tier 3 Fallback) failed: {error}")
                        job_data["tier_3_error"] = str(error)
                        job_data["tier_3_success"] = False
                services_attempted.append(f"{TYPEMYWORDZ2_NAME}_tier3")
            
            elif tier_3_service == "deepgram":
                if not DEEPGRAM_SERVICE_RAILWAY_URL:
                    logger.error(f"{DEEPGRAM_NAME} Service URL not configured, skipping Tier 3 for job {job_id}")
                    job_data["tier_3_error"] = f"{DEEPGRAM_NAME} Service URL not configured"
                else:
                    try:
                        logger.info(f"🔄 Attempting {DEEPGRAM_NAME} (Tier 3 Fallback) for job {job_id}")
                        if compressed_path is None:
                            compressed_path, compression_stats = compress_audio_for_transcription(tmp_path, job_id=job_id)
                            logger.info(f"Audio compressed for {DEEPGRAM_NAME}: {compression_stats}")
                        transcription_result = await transcribe_with_deepgram(compressed_path, language_code, speaker_labels_enabled, job_id)
                        job_data["tier_3_used"] = "deepgram"
                        job_data["tier_3_success"] = True
                    except Exception as error:
                        logger.error(f"❌ {DEEPGRAM_NAME} (Tier 3 Fallback) failed: {error}")
                        job_data["tier_3_error"] = str(error)
                        job_data["tier_3_success"] = False
                services_attempted.append(f"{DEEPGRAM_NAME}_tier3")
            
        check_cancellation()

        # Final result processing
        if not transcription_result or transcription_result.get("status") == "failed":
            logger.error(f"❌ All transcription services failed for job {job_id}. Services attempted: {services_attempted}")
            job_data.update({
                "status": "failed",
                "error": "Transcription failed after multiple attempts. Please try again later.",
                "completed_at": datetime.now().isoformat(),
                "services_attempted": services_attempted
            })
        else:
            logger.info(f"✅ Transcription completed successfully for job {job_id}")
            service_used_name = (job_data.get("tier_1_used") or job_data.get("tier_2_used") or job_data.get("tier_3_used"))
            model_used = "N/A"

            if service_used_name == "assemblyai":
                model_used = (transcription_result.get("model_used")
                              or job_data.get("assemblyai_model", "unknown"))
            elif service_used_name == "openai_whisper":
                model_used = "whisper-1"
            elif service_used_name == "deepgram":
                model_used = "deepgram-nova"

            logger.info(f"📊 Job {job_id} for user {user_email} completed. Service: {service_used_name}, Model: {model_used}")

            job_data.update({
                "status": "completed",
                "transcription": transcription_result["transcription"],
                "language": transcription_result.get("language", language_code),
                "completed_at": datetime.now().isoformat(),
                "word_count": transcription_result.get("word_count", 0),
                "duration_seconds": transcription_result.get("duration", 0),
                "speaker_labels": speaker_labels_enabled,
                "service_used": service_used_name,
                "model_used": model_used,
                "services_attempted": services_attempted,
                # Word timings, when the service that ran provides them.
                # Absent for services that do not, and the editor copes.
                "segments": transcription_result.get("segments") or [],
                "timings_source": transcription_result.get("timings_source")
            })

            # Charge for the audio we actually transcribed, by the minute,
            # rounded up. Charged on success only: a failed job is free.
            billed_seconds = (transcription_result.get("duration")
                              or (duration_minutes or 0) * 60)
            cost = minutes_to_credits(billed_seconds)
            charge = await charge_credits(
                job_data.get("user_id") or "", user_email, cost, f"transcription {job_id}"
            )
            job_data["credits_charged"] = charge.get("charged", 0)
            job_data["credits_remaining"] = charge.get("remaining")

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
            "error": "An unexpected error occurred during transcription. Please try again later.",
            "completed_at": datetime.now().isoformat()
        })
    finally:
        # Clean up original temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"Cleaned up original temp file: {tmp_path}")
        
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
    human_expiry_task = asyncio.create_task(human_expiry_monitor())
    human_payout_task = asyncio.create_task(human_payout_monitor())
    logger.info("Human worker TAT and payout monitor tasks created")
    yield
    logger.info("Application lifespan shutdown")
    health_task.cancel()
    human_expiry_task.cancel()
    human_payout_task.cancel()
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
app = FastAPI(title=f"Enhanced Transcription Service with {TYPEMYWORDZ1_NAME}, {TYPEMYWORDZ2_NAME}, {DEEPGRAM_NAME}, {TYPEMYWORDZ_AI_NAME} & Google Gemini", lifespan=lifespan)
logger.info("FastAPI app created successfully")

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
        "message": f"Enhanced Transcription Service with {TYPEMYWORDZ1_NAME}, {TYPEMYWORDZ2_NAME}, {DEEPGRAM_NAME}, {TYPEMYWORDZ_AI_NAME} & Google Gemini is running!",
        "features": [
            f"AssemblyAI integration with smart model selection",
            f"OpenAI Whisper integration for transcription",
            f"Deepgram integration for transcription",
            "Three-tier automatic fallback between services",
            "Paystack payment integration",
            f"Speaker diarization for AssemblyAI and Deepgram",
            "Language selection for transcription",
            f"User-driven AI features (summarization, Q&A, and bullet points) via TypeMyworDz AI (Anthropic)",
            f"Admin-driven AI formatting via TypeMyworDz AI (Anthropic) and Google Gemini",
            "Google Gemini integration for AI queries - NOW AVAILABLE FOR ALL PAID USERS"
        ],
        "logic": {
            "free_user_transcription": f"Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
            "three_day_plan_transcription": f"Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
            "one_week_plan_transcription": f"Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
            "monthly_plan_transcription": f"Primary={TYPEMYWORDZ2_NAME} → Fallback1={TYPEMYWORDZ1_NAME} → Fallback2={DEEPGRAM_NAME}",
            "yearly_plan_transcription": f"Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
            "admin_transcription": f"Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
            "speaker_labels_transcription": f"Always use {TYPEMYWORDZ1_NAME} first → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
            "openai_tester_transcription": f"Always use {TYPEMYWORDZ2_NAME} (no fallback for {OPENAI_TESTER_EMAIL})",
            "deepgram_tester_transcription": f"Primary=Deepgram → Fallback=OpenAI for {DEEPGRAM_TESTER_EMAIL}",
            "assemblyai_models": f"{TYPEMYWORDZ1_NAME} universal-3-5-pro, falling back to universal-2 for other languages",
            "ai_features_access": "Only for Three-Day, One-Week, Monthly Plan, and Yearly Plan plans",
            "gemini_access": "NOW AVAILABLE FOR ALL PAID AI USERS (Three-Day, One-Week, Monthly Plan, Yearly Plan plans)",
            "assemblyai": f"TypeMyworDz1 (AssemblyAI)",
            "openai_whisper": f"TypeMyworDz2 (OpenAI Whisper-1)",
            "deepgram": f"Deepgram",
            "anthropic_ai": f"TypeMyworDz AI (Anthropic Claude)",
            "google_gemini_ai": "Google Gemini - Available for ALL paid AI users",
            "admin_emails": ADMIN_EMAILS,
            "openai_tester_email": OPENAI_TESTER_EMAIL,
            "deepgram_tester_email": DEEPGRAM_TESTER_EMAIL
        },
        "stats": {
            "active_jobs": len(jobs),
            "background_tasks": len(active_background_tasks),
            "cancellation_flags": len(cancellation_flags)
        }
    }

@app.post("/api/trainee/register")
async def register_trainee(request: Request):
    actor = await _trainee_actor(request)
    payload = await request.json()
    official_name = str(payload.get("official_name") or "").strip()
    country_code = str(payload.get("country_code") or "").strip().upper()
    if len(official_name) < 2:
        raise HTTPException(status_code=400, detail="Enter your official ID name.")
    registered_name = str((actor.get("profile") or {}).get("officialIdName") or "").strip()
    if registered_name and registered_name.casefold() != official_name.casefold():
        raise HTTPException(status_code=409, detail="Your official name is locked from trainee registration. Contact support if it needs correcting.")
    if country_code != TRAINEE_COUNTRY:
        raise HTTPException(status_code=400, detail="Training enrollment is currently limited to Kenya.")
    if not db:
        raise HTTPException(status_code=503, detail="The workflow database is unavailable.")
    updates = {
        "name": official_name,
        "officialIdName": official_name,
        "country": "Kenya",
        "countryCode": TRAINEE_COUNTRY,
        "traineeStatus": "payment_pending",
        "trainingPaymentStatus": "pending",
        "trainingRoomAccess": False,
        "workerApproved": False,
        "traineeAccountPendingDeletion": bool(payload.get("created_for_trainee")),
        "traineeAppliedAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    await asyncio.to_thread(db.collection("users").document(actor["uid"]).set, updates, merge=True)
    return {"status": "payment_pending", "country": "Kenya", "fee_usd": TRAINEE_PRICE_USD}



@app.post("/api/trainee/cancel-pending")
async def cancel_pending_trainee(request: Request):
    """Remove a trainee auth/profile created for a checkout that did not pay."""
    actor = await _trainee_actor(request)
    profile = actor["profile"]
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if profile.get("trainingPaymentStatus") == "paid" or profile.get("trainingRoomAccess"):
        raise HTTPException(status_code=409, detail="A paid trainee account cannot be cancelled this way.")
    delete_auth = bool(profile.get("traineeAccountPendingDeletion") or payload.get("created_for_trainee") or payload.get("force_pending_cleanup"))
    deleted = {"auth": False, "profile": False}
    if delete_auth:
        try:
            await asyncio.to_thread(firebase_auth.delete_user, actor["uid"])
            deleted["auth"] = True
        except firebase_auth.UserNotFoundError:
            deleted["auth"] = True
        except Exception as exc:
            logger.warning("Could not delete pending trainee auth account %s: %s", actor["uid"], exc)
        if db:
            await asyncio.to_thread(db.collection("users").document(actor["uid"]).delete)
            deleted["profile"] = True
    elif db:
        # An existing client account may have started trainee enrollment. Keep
        # that normal account, but remove every pending-trainee marker so a
        # failed checkout cannot change its access or trap it in Training Room.
        await asyncio.to_thread(db.collection("users").document(actor["uid"]).set, {
            "traineeStatus": "not_started",
            "trainingPaymentStatus": "not_submitted",
            "trainingRoomAccess": False,
            "traineeAccountPendingDeletion": False,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }, merge=True)
        deleted["profile"] = True
    logger.info("Removed unpaid trainee enrollment for %s (auth_deleted=%s)", actor["email"], delete_auth)
    return {"success": True, "deleted": deleted}


@app.post("/api/initialize-trainee-payment")
async def initialize_trainee_payment(request: Request):
    """Start trainee checkout before Firebase signup; no account exists yet."""
    payload = await request.json()
    email = str(payload.get("email") or "").strip().lower()
    official_name = str(payload.get("official_name") or "").strip()
    country_code = str(payload.get("country_code") or "").strip().upper()
    provider = str(payload.get("provider") or "paystack").strip().lower()
    callback_url = str(payload.get("callback_url") or f"{APP_URL}/trainee-signup?payment=success&trainee=1")
    if "@" not in email or len(official_name) < 2:
        raise HTTPException(status_code=400, detail="A valid email and official ID name are required.")
    if country_code != TRAINEE_COUNTRY:
        raise HTTPException(status_code=400, detail="Training enrollment is currently limited to Kenya.")
    if provider not in {"paystack", "kora"}:
        raise HTTPException(status_code=400, detail="That payment option is not available.")
    if provider == "paystack" and not PAYSTACK_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Paystack checkout is not configured yet.")
    if provider == "kora" and not KORA_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Kora checkout is not configured yet.")

    local_amount, local_currency = get_local_amount_and_currency(TRAINEE_PRICE_USD, TRAINEE_COUNTRY, TRAINEE_PRODUCT)
    reference = ("tmw-trainee-" if provider == "kora" else "tmw-") + uuid.uuid4().hex
    metadata = {
        "product": TRAINEE_PRODUCT,
        "official_name": official_name,
        "country_code": TRAINEE_COUNTRY,
        "trainee_signup": "true",
    }
    if provider == "paystack":
        response = requests.post(
            "https://api.paystack.co/transaction/initialize",
            headers={"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}", "Content-Type": "application/json"},
            json={
                "email": email,
                "amount": int(round(local_amount * 100)),
                "currency": local_currency,
                "reference": reference,
                "callback_url": callback_url,
                "channels": get_payment_channels(TRAINEE_COUNTRY, TRAINEE_PRODUCT),
                "metadata": metadata,
            },
            timeout=15,
        )
        data = response.json()
        if response.status_code >= 400 or not data.get("status"):
            raise HTTPException(status_code=502, detail=data.get("message") or "Paystack checkout could not be started.")
        checkout_url = (data.get("data") or {}).get("authorization_url")
    else:
        redirect_url = callback_url
        separator = "&" if "?" in redirect_url else "?"
        redirect_url = f"{redirect_url}{separator}reference={reference}&kora=success&trainee=1"
        response = requests.post(
            "https://api.korapay.com/merchant/api/v1/charges/initialize",
            headers={"Authorization": f"Bearer {KORA_SECRET_KEY}", "Content-Type": "application/json"},
            json={
                "amount": int(round(local_amount)),
                "currency": local_currency,
                "reference": reference,
                "redirect_url": redirect_url,
                "notification_url": KORA_NOTIFICATION_URL or f"{BACKEND_PUBLIC_URL}/api/kora-webhook",
                "narration": "TypeMyworDz Training Room enrollment",
                "channels": ["mobile_money", "card"],
                "customer": {"email": email, "name": official_name},
                "metadata": metadata,
            },
            timeout=15,
        )
        data = response.json()
        if response.status_code >= 400 or not data.get("status"):
            raise HTTPException(status_code=502, detail=data.get("message") or "Kora checkout could not be started.")
        checkout_url = (data.get("data") or {}).get("checkout_url")
    if not checkout_url:
        raise HTTPException(status_code=502, detail="The payment provider did not return a checkout link.")
    if db:
        await asyncio.to_thread(db.collection("payment_intents").document(reference).set, {
            "uid": None,
            "email": email,
            "product": TRAINEE_PRODUCT,
            "provider": provider,
            "countryCode": TRAINEE_COUNTRY,
            "officialName": official_name,
            "amountUsd": TRAINEE_PRICE_USD,
            "amountLocal": local_amount,
            "currency": local_currency,
            "status": "pending",
            "createdAt": firestore.SERVER_TIMESTAMP,
        })
    return {"status": True, "authorization_url": checkout_url, "checkout_url": checkout_url, "reference": reference, "local_amount": local_amount, "local_currency": local_currency}

@app.post("/api/initialize-kora-trainee-payment")
async def initialize_kora_trainee_payment(request: Request):
    actor = await _trainee_actor(request)
    payload = await request.json()
    official_name = str(payload.get("official_name") or actor["profile"].get("officialIdName") or actor["profile"].get("name") or "").strip()
    country_code = str(payload.get("country_code") or actor["profile"].get("countryCode") or "").upper()
    if country_code != TRAINEE_COUNTRY:
        raise HTTPException(status_code=400, detail="Training enrollment is currently limited to Kenya.")
    if not KORA_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Kora checkout is not configured yet.")
    local_amount, local_currency = get_local_amount_and_currency(TRAINEE_PRICE_USD, TRAINEE_COUNTRY, "trainee-training")
    reference = "tmw-trainee-" + uuid.uuid4().hex
    notification_url = KORA_NOTIFICATION_URL or f"{BACKEND_PUBLIC_URL}/api/kora-webhook"
    redirect_url = str(payload.get("redirect_url") or f"{APP_URL}/?kora=success")
    kora_payload = {
        "amount": int(round(local_amount)),
        "currency": local_currency,
        "reference": reference,
        "redirect_url": redirect_url,
        "notification_url": notification_url,
        "narration": "TypeMyworDz Training Room enrollment",
        "channels": ["mobile_money", "card"],
        "customer": {"email": actor["email"], "name": official_name},
        "metadata": {"product": TRAINEE_PRODUCT, "user-id": actor["uid"], "country": TRAINEE_COUNTRY},
    }
    headers = {"Authorization": f"Bearer {KORA_SECRET_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.post("https://api.korapay.com/merchant/api/v1/charges/initialize", headers=headers, json=kora_payload, timeout=15)
        data = response.json()
        if response.status_code >= 400 or not data.get("status"):
            raise HTTPException(status_code=502, detail=data.get("message") or "Kora checkout could not be started.")
        if db:
            await asyncio.to_thread(db.collection("payment_intents").document(reference).set, {
                "uid": actor["uid"], "email": actor["email"], "product": TRAINEE_PRODUCT, "provider": "kora",
                "countryCode": TRAINEE_COUNTRY, "amountUsd": TRAINEE_PRICE_USD, "amountLocal": local_amount,
                "currency": local_currency, "status": "pending", "createdAt": firestore.SERVER_TIMESTAMP,
            })
        return {"status": True, "checkout_url": data.get("data", {}).get("checkout_url"), "reference": reference, "local_amount": local_amount, "local_currency": local_currency}
    except HTTPException:
        raise
    except requests.RequestException as exc:
        logger.exception("Kora initialization failed")
        raise HTTPException(status_code=502, detail="Kora checkout could not be reached.") from exc


async def verify_kora_and_enroll(reference: str):
    """Verify a live Kora charge and fulfil either a trainee, plan, or top-up intent."""
    if not KORA_SECRET_KEY:
        return {"success": False, "error": "Kora configuration missing"}
    response = requests.get(
        f"https://api.korapay.com/merchant/api/v1/charges/{reference}",
        headers={"Authorization": f"Bearer {KORA_SECRET_KEY}"},
        timeout=15,
    )
    data = response.json()
    charge = data.get("data") or {}
    if response.status_code >= 400 or not data.get("status") or str(charge.get("status") or "").lower() != "success":
        return {"success": False, "error": data.get("message") or "Kora payment is not successful yet."}
    if not db:
        return {"success": False, "error": "Firestore not initialized"}
    intent_snap = await asyncio.to_thread(db.collection("payment_intents").document(reference).get)
    intent = intent_snap.to_dict() if intent_snap.exists else {}
    product = str(intent.get("product") or "")
    if not intent or not product:
        return {"success": False, "error": "Unknown Kora payment reference"}

    if intent.get("status") == "paid":
        return {"success": True, "already_applied": True, "plan": product, "trainee_enrolled": product == TRAINEE_PRODUCT}

    email = intent.get("email")
    country_code = str(intent.get("countryCode") or "").upper()
    currency = str(charge.get("currency") or intent.get("currency") or "KES")
    amount_usd = float(intent.get("amountUsd") or price_for(product, country_code) or 0)
    if product == TRAINEE_PRODUCT:
        if not await get_user_profile_by_email_firestore(email):
            await asyncio.to_thread(db.collection("payment_intents").document(reference).set, {
                "status": "paid", "paidAt": firestore.SERVER_TIMESTAMP,
                "amountUsd": TRAINEE_PRICE_USD, "currency": currency,
            }, merge=True)
            return {"success": True, "requires_account": True, "email": email, "trainee_enrolled": False, "plan": product}
        result = await enroll_paid_trainee(email, reference, TRAINEE_PRICE_USD, currency, country_code, "kora")
    else:
        result = await update_user_credits_paystack(
            email=email,
            plan_name=product,
            amount=amount_usd,
            currency=currency,
            update_admin_revenue=bool(intent.get("updateAdminRevenue")),
            country_code=country_code,
            reference=reference,
        )
    if result.get("success"):
        result["plan"] = product
        await asyncio.to_thread(
            db.collection("payment_intents").document(reference).set,
            {"status": "paid", "verifiedAt": firestore.SERVER_TIMESTAMP},
            merge=True,
        )
    return result


@app.post("/api/initialize-kora-payment")
async def initialize_kora_payment(request: PaystackInitializationRequest):
    """Initialize a Kora checkout for an African plan or credit top-up."""
    if not KORA_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Kora checkout is not configured yet.")
    if not request.email or not request.user_id:
        raise HTTPException(status_code=400, detail="A signed-in account is required.")
    base_usd = price_for(request.plan_name, request.country_code)
    if base_usd is None:
        raise HTTPException(status_code=400, detail="That is not something we sell.")
    local_amount, local_currency = get_local_amount_and_currency(base_usd, request.country_code, request.plan_name)
    reference = "tmw-" + uuid.uuid4().hex
    redirect_url = str(request.callback_url or f"{APP_URL}/?kora=success")
    separator = "&" if "?" in redirect_url else "?"
    redirect_url = f"{redirect_url}{separator}reference={reference}"
    notification_url = KORA_NOTIFICATION_URL or f"{BACKEND_PUBLIC_URL}/api/kora-webhook"
    kora_payload = {
        "amount": int(round(local_amount)),
        "currency": local_currency,
        "reference": reference,
        "redirect_url": redirect_url,
        "notification_url": notification_url,
        "narration": f"TypeMyworDz {request.plan_name}",
        "channels": ["mobile_money", "card"],
        "customer": {"email": request.email},
        "metadata": {
            "product": request.plan_name,
            "user-id": request.user_id,
            "country": request.country_code,
        },
    }
    headers = {"Authorization": f"Bearer {KORA_SECRET_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.post(
            "https://api.korapay.com/merchant/api/v1/charges/initialize",
            headers=headers,
            json=kora_payload,
            timeout=15,
        )
        data = response.json()
        if response.status_code >= 400 or not data.get("status"):
            raise HTTPException(status_code=502, detail=data.get("message") or "Kora checkout could not be started.")
        if db:
            await asyncio.to_thread(
                db.collection("payment_intents").document(reference).set,
                {
                    "uid": request.user_id,
                    "email": request.email,
                    "product": request.plan_name,
                    "provider": "kora",
                    "countryCode": request.country_code,
                    "amountUsd": base_usd,
                    "amountLocal": local_amount,
                    "currency": local_currency,
                    "updateAdminRevenue": bool(request.update_admin_revenue),
                    "status": "pending",
                    "createdAt": firestore.SERVER_TIMESTAMP,
                },
            )
        return {
            "status": True,
            "checkout_url": data.get("data", {}).get("checkout_url"),
            "reference": reference,
            "local_amount": local_amount,
            "local_currency": local_currency,
        }
    except HTTPException:
        raise
    except requests.RequestException as exc:
        logger.exception("Kora initialization failed")
        raise HTTPException(status_code=502, detail="Kora checkout could not be reached.") from exc


@app.post("/api/verify-kora-payment")
async def verify_kora_payment(request: KoraVerificationRequest):
    result = await verify_kora_and_enroll(request.reference)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Kora payment verification failed."))
    return {"status": "success", "data": {"plan": result.get("plan", ""), "reference": request.reference, "training_room": bool(result.get("trainee_enrolled"))}}


@app.post("/api/kora-webhook")
async def kora_webhook(request: Request):
    payload = await request.json()
    if payload.get("event") not in (None, "charge.success"):
        return {"status": "received"}
    reference = str((payload.get("data") or {}).get("reference") or "").strip()
    if reference:
        result = await verify_kora_and_enroll(reference)
        if not result.get("success"):
            logger.warning("Kora webhook received but verification was not successful: %s", result.get("error"))
    return {"status": "received"}


@app.post("/api/initialize-paystack-payment")
async def initialize_paystack_payment(request: PaystackInitializationRequest):
    logger.info(f"Initializing Paystack payment for {request.email} in {request.country_code}: Base USD {request.amount}")
    
    if not PAYSTACK_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Paystack configuration missing")
    
    try:
        # The price comes from the server's own table. Whatever amount the
        # browser sent is ignored, so the price cannot be edited on the way in.
        base_usd = price_for(request.plan_name, request.country_code)
        if base_usd is None:
            raise HTTPException(status_code=400, detail="That is not something we sell.")

        local_amount, local_currency = get_local_amount_and_currency(base_usd, request.country_code, request.plan_name)
        payment_channels = get_payment_channels(request.country_code, request.plan_name)

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
                'base_usd_amount': base_usd,
                'country_code': request.country_code,
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
                    },
                    {
                        'display_name': "Update Admin Revenue",
                        'variable_name': "update_admin_revenue",
                        'value': str(request.update_admin_revenue)
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
            reference = verification_result['reference']
            
            # Extract base_usd_amount and country_code from raw_data metadata
            base_usd_amount = verification_result['raw_data'].get('metadata', {}).get('base_usd_amount')
            country_code = verification_result['raw_data'].get('metadata', {}).get('country_code')
            update_admin_revenue_flag = verification_result['raw_data'].get('metadata', {}).get('update_admin_revenue', 'False').lower() == 'true'

            # Trainee checkout is payment-first. If the account does not exist
            # yet, record the paid intent and let the browser create the account
            # only after Paystack has confirmed success.
            if plan_name == TRAINEE_PRODUCT and not await get_user_profile_by_email_firestore(email):
                if db:
                    await asyncio.to_thread(db.collection('payment_intents').document(reference).set, {
                        'email': email, 'product': TRAINEE_PRODUCT, 'provider': 'paystack',
                        'countryCode': country_code or TRAINEE_COUNTRY,
                        'officialName': verification_result['raw_data'].get('metadata', {}).get('official_name') or '',
                        'amountUsd': TRAINEE_PRICE_USD, 'currency': currency,
                        'status': 'paid', 'paidAt': firestore.SERVER_TIMESTAMP,
                    }, merge=True)
                return {"status": "success", "message": "Payment verified. Create the trainee account to continue.", "data": {"amount": amount, "currency": currency, "email": email, "plan": plan_name, "reference": reference, "trainee_pending_account": True}}

            # Pass base_usd_amount, country_code, and update_admin_revenue_flag
            credit_result = await update_user_credits_paystack(email, plan_name, base_usd_amount or amount, currency, update_admin_revenue_flag, country_code, reference) 
            
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
                logger.warning(f"⚠️ Payment verified but credit update failed for {email}: {credit_result.get('error')}")
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
        
        # Optional: Verify webhook signature for production.
        # if PAYSTACK_WEBHOOK_SECRET:
        #     import hmac
        #     import hashlib
        #     expected_signature = hmac.new(PAYSTACK_WEBHOOK_SECRET.encode('utf-8'), body, hashlib.sha512).hexdigest()
        #     if not hmac.compare_digest(expected_signature, signature):
        #         logger.warning("❌ Webhook signature mismatch!")
        #         raise HTTPException(status_code=400, detail="Invalid webhook signature")
        
        event_type = webhook_data.get('event')
        logger.info(f"Processing Paystack webhook event: {event_type}")
        
        if event_type == 'charge.success':
            data = webhook_data.get('data', {})
            customer_email = data.get('customer', {}).get('email')
            amount = data.get('amount', 0) / 100
            currency = data.get('currency')
            reference = data.get('reference')
            plan_name = data.get('metadata', {}).get('plan', 'Unknown')
            base_usd_amount = data.get('metadata', {}).get('base_usd_amount')
            country_code = data.get('metadata', {}).get('country_code')
            update_admin_revenue_flag = data.get('metadata', {}).get('update_admin_revenue', 'False').lower() == 'true'

            logger.info(f"🔔 Webhook: Payment successful - {customer_email} paid {amount} {currency} for {plan_name}. Base USD: {base_usd_amount}, Country: {country_code}, Update Revenue: {update_admin_revenue_flag}")
            
            if customer_email:
                credit_result = await update_user_credits_paystack(customer_email, plan_name, base_usd_amount or amount, currency, update_admin_revenue_flag, country_code, reference)
                if credit_result['success']:
                    logger.info(f"✅ Webhook: Credits updated automatically for {customer_email} in Firestore.")
                else:
                    logger.warning(f"⚠️ Webhook: Failed to update credits for {customer_email} in Firestore: {credit_result.get('error')}")
            
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
        "kora_configured": bool(KORA_SECRET_KEY),
        "public_key_configured": bool(PAYSTACK_PUBLIC_KEY),
        "webhook_secret_configured": bool(PAYSTACK_WEBHOOK_SECRET),
        "assemblyai_configured": bool(ASSEMBLYAI_API_KEY),
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_whisper_service_configured": bool(OPENAI_WHISPER_SERVICE_RAILWAY_URL),
        "google_gemini_configured": bool(GEMINI_API_KEY),
        "deepgram_service_configured": bool(DEEPGRAM_SERVICE_RAILWAY_URL),
        "admin_emails": ADMIN_EMAILS,
        "openai_tester_email": OPENAI_TESTER_EMAIL,
        "deepgram_tester_email": DEEPGRAM_TESTER_EMAIL,
        "gemini_access": "NOW AVAILABLE FOR ALL PAID AI USERS (Three-Day, One-Week, Monthly Plan, Yearly Plan plans)",
        "endpoints": {
            "initialize_payment": "/api/initialize-paystack-payment",
            "verify_payment": "/api/verify-payment",
            "webhook": "/api/paystack-webhook",
            "status": "/api/paystack-status",
            "transcribe": "/transcribe",
            "ai_user_query": "/ai/user-query",
            "ai_user_query_gemini": "/ai/user-query-gemini",
            "ai_ask": "/ai/ask",
            "ai_admin_format": "/ai/admin-format",
            "ai_admin_format_gemini": "/ai/admin-format-gemini",
        },
        "supported_currencies": ["NGN", "USD", "GHS", "ZAR", "KES"],
        "supported_plans": [
            "Three-Day Plan",
            "One-Week Plan",
            "Monthly Plan",
            "Yearly Plan"
        ],
        "conversion_rates_usd_to_local": USD_TO_LOCAL_RATES
    }

@app.get("/api/list-gemini-models")
async def list_gemini_models():
    logger.info("Listing available Gemini models...")
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Google Gemini service is not initialized (API key missing or invalid).")
    
    try:
        models = genai.list_models()
        gemini_models_info = []
        for m in models:
            if 'gemini' in m.name and 'generateContent' in m.supported_generation_methods:
                gemini_models_info.append({
                    "name": m.name,
                    "display_name": m.display_name,
                    "version": m.version,
                    "supported_generation_methods": m.supported_generation_methods,
                    "input_token_limit": m.input_token_limit,
                    "output_token_limit": m.output_token_limit
                })
        logger.info(f"Found {len(gemini_models_info)} Gemini models.")
        return {"available_gemini_models": gemini_models_info}
    except Exception as e:
        logger.error(f"Error listing Gemini models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list Gemini models: {str(e)}")


# ===================== Ask TypeMyworDz: reading attachments ================
# Clients can attach images, PDFs and Word documents to a question. The models
# cannot open a .docx or a .pdf, so anything that is not an image is turned
# into plain text here first. Images are passed through as base64 for the
# model's vision input.

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS_PER_FILE = 200000
MAX_ATTACHMENTS = 8
IMAGE_TYPES = {
    'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
    'gif': 'image/gif', 'webp': 'image/webp',
}


def _file_ext(name: str) -> str:
    return (name or '').rsplit('.', 1)[-1].lower() if '.' in (name or '') else ''


def read_attachment(filename: str, raw: bytes):
    """Turn one uploaded file into something a model can read.

    Returns {'kind': 'text'|'image'|'error', ...}. Errors carry a plain-English
    message that is shown to the client rather than swallowed, so that an
    unreadable file never looks like the assistant simply ignored it.
    """
    name = filename or 'file'
    ext = _file_ext(name)

    if not raw:
        return {'kind': 'error', 'name': name, 'message': 'the file was empty'}
    if len(raw) > MAX_ATTACHMENT_BYTES:
        return {'kind': 'error', 'name': name, 'message': 'the file is larger than 20 MB'}

    if ext in IMAGE_TYPES:
        return {'kind': 'image', 'name': name, 'media_type': IMAGE_TYPES[ext],
                'data': base64.b64encode(raw).decode('ascii')}

    if ext == 'pdf':
        try:
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(raw))
            if getattr(reader, 'is_encrypted', False):
                try:
                    reader.decrypt('')
                except Exception:
                    return {'kind': 'error', 'name': name, 'message': 'the PDF is password protected'}
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or '')
                except Exception:
                    pages.append('')
            text = '\n\n'.join(p for p in pages if p.strip())
            if not text.strip():
                return {'kind': 'error', 'name': name,
                        'message': 'this PDF has no text in it, it looks like a scan. Try attaching it as an image instead'}
            return {'kind': 'text', 'name': name, 'text': text[:MAX_TEXT_CHARS_PER_FILE]}
        except Exception as e:
            return {'kind': 'error', 'name': name, 'message': f'the PDF could not be read ({e})'}

    if ext in ('docx', 'doc'):
        try:
            doc = Document(BytesIO(raw))
            parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                    if cells:
                        parts.append(' | '.join(cells))
            text = '\n'.join(parts)
            if not text.strip():
                return {'kind': 'error', 'name': name, 'message': 'the document had no text in it'}
            return {'kind': 'text', 'name': name, 'text': text[:MAX_TEXT_CHARS_PER_FILE]}
        except Exception:
            return {'kind': 'error', 'name': name,
                    'message': 'the document could not be read. If it is an older .doc file, save it as .docx and try again'}

    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = raw.decode('latin-1')
        except Exception:
            return {'kind': 'error', 'name': name, 'message': 'this file type is not supported'}
    if '\x00' in text[:2000]:
        return {'kind': 'error', 'name': name, 'message': 'this file type is not supported'}
    return {'kind': 'text', 'name': name, 'text': text[:MAX_TEXT_CHARS_PER_FILE]}


def parse_history(raw_history: str):
    """Prior turns of the conversation, sent by the browser as JSON.

    Anything malformed is ignored rather than failing the request: losing the
    earlier context is a far better outcome than losing the question.
    """
    if not raw_history:
        return []
    try:
        data = json.loads(raw_history)
    except Exception:
        logger.warning("Ask: history was not valid JSON, continuing without it.")
        return []
    if not isinstance(data, list):
        return []
    out = []
    for turn in data[-40:]:
        if not isinstance(turn, dict):
            continue
        role = turn.get('role')
        content = turn.get('content')
        if role in ('user', 'assistant') and isinstance(content, str) and content.strip():
            out.append({'role': role, 'content': content.strip()})
    # A conversation has to start with the client and alternate; drop any
    # leading assistant turns that would make the API reject the request.
    while out and out[0]['role'] != 'user':
        out.pop(0)
    return out


# The assistant is used in two completely different places and must not
# confuse them. Beside a transcript it is a transcript assistant. On the
# standalone Ask page it is a general assistant that happens to be ours, and
# must never mention transcripts, offer to transcribe anything, or ask which
# transcript the client means. Sending one prompt for both was making the
# research page answer as though a transcript were sitting in front of it.

_ASK_SHARED = (
    "Answer clearly and directly, in plain language, without padding or "
    "flattery. Do not use emoji. "
    "Attachments: when the client attaches a document, its full text is placed "
    "into the message between '--- Attached file: NAME ---' and "
    "'--- end of NAME ---' markers, and images are attached directly. Anything "
    "arriving that way is genuinely attached, so read it and answer from it. "
    "Never tell the client you cannot see or access an attached file, and never "
    "ask them to paste its contents. "
    "Never describe your own situation or setup to the user. Do not begin with "
    "preamble about what you were or were not given, such as 'Based on general "
    "knowledge' or 'Here is'. Simply answer the question. "
    "Formatting: you may use **bold**, bullet lines beginning with '- ', and "
    "numbered lines beginning with '1. '. Do not use tables, headings marked "
    "with '#', code fences, or single asterisks for emphasis."
)

ASK_SYSTEM_PROMPT_TRANSCRIPT = (
    "You are TypeMyworDz Assistant, helping a client work with a transcript "
    "they have just had made. Base your answer on the transcript you have been "
    "given, and say plainly when something is not in it rather than guessing. "
    + _ASK_SHARED
)

ASK_SYSTEM_PROMPT_GENERAL = (
    "You are TypeMyworDz Assistant, a general assistant that helps with "
    "whatever the client is working on: research, writing, planning, analysis, "
    "code, study, and everyday questions. "
    "There is no transcript in this conversation and none is expected. Never "
    "mention transcripts, never offer to transcribe or caption anything, and "
    "never ask the client which transcript or recording they mean. If a "
    "question is too vague to answer, ask what they would like you to look at, "
    "without assuming it is a transcript. "
    + _ASK_SHARED
)

# Kept so that anything still referring to the old single prompt keeps working.
ASK_SYSTEM_PROMPT = ASK_SYSTEM_PROMPT_GENERAL


@app.post("/ai/user-query")
async def ai_user_query(
    transcript: str = Form(...),
    user_prompt: str = Form(...),
    model: str = Form("claude-haiku-4-5-20251001"),
    max_tokens: int = Form(1000),
    user_plan: str = Form("free"),
    user_email: str = Form("")
):
    logger.info(f"AI user query endpoint called. Model: {model}, Prompt: '{user_prompt}', User Plan: {user_plan}")

    has_credits = await account_has_usable_credits(user_email=user_email)
    if not is_ai_allowed(user_plan, user_email, has_credits):
        raise HTTPException(status_code=403, detail="AI Assistant features are only available for paid AI users (Three-Day, One-Week, Monthly Plan, Yearly Plan plans). Please upgrade your plan.")

    if not claude_client:
        raise HTTPException(status_code=503, detail=f"{TYPEMYWORDZ_AI_NAME} service is not initialized (API key missing or invalid).")

    try:
        if len(transcript) > 100000:
            raise HTTPException(status_code=400, detail="Transcript is too long. Please use a shorter transcript.")
        

        full_prompt = f"{user_prompt}\n\nHere is the transcript:\n{transcript}"

        message = claude_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            timeout=30.0,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        ai_response = claude_text(message)
        if not ai_response:
            ai_response = (
                "That answer came back empty, which is a hiccup on the model's "
                "side rather than anything you did. Please ask again."
            )
        logger.info(f"Successfully processed AI user query with {model}.")
        return {"ai_response": ai_response}

    except anthropic.APIError as e:
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

@app.get("/ai/models")
async def ai_models(user_plan: str = "free", user_email: str = "", has_transcript: str = "true"):
    """The models this caller is allowed to pick from.

    The Settings page calls this so that the list a client sees always matches
    what the server will actually accept.
    """
    want_transcript_models = str(has_transcript).strip().lower() not in ("false", "0", "no")
    has_credits = await account_has_usable_credits(user_email=user_email)
    allowed = ask_models_for(user_plan, user_email, want_transcript_models, has_credits)
    locked = ask_models_locked_for(user_plan, user_email, want_transcript_models, has_credits)
    resolved_default = None
    if allowed:
        resolved_default, _ = resolve_ask_model("", user_plan, user_email, want_transcript_models, has_credits)
    return {
        "models": allowed,
        "locked": locked,
        "default": resolved_default,
        "premium_included": any(m["tier"] == "premium" for m in allowed),
    }



# ---------------------------------------------------------------------------
# Talking to the model providers.
#
# OpenAI and Mistral both speak the OpenAI chat format, so one function covers
# both. Gemini is called over plain HTTP rather than through the SDK, because
# the installed SDK cannot switch thinking off or report when an answer was
# cut short, and both of those matter here.
# ---------------------------------------------------------------------------

OPENAI_FORMAT_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/chat/completions", lambda: OPENAI_API_KEY),
    "mistral": ("https://api.mistral.ai/v1/chat/completions", lambda: MISTRAL_API_KEY),
}


def _ask_openai_format(provider, model_id, system_prompt, turns, question, images, max_tokens):
    """Call OpenAI or Mistral and return the answer text."""
    url, get_key = OPENAI_FORMAT_ENDPOINTS[provider]
    key = get_key()
    if not key:
        raise HTTPException(status_code=503, detail=f"{TYPEMYWORDZ_AI_NAME} is not connected to that model right now.")

    content = []
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img['media_type']};base64,{img['data']}"},
        })
    content.append({"type": "text", "text": question})

    messages = [{"role": "system", "content": system_prompt}]
    for turn in turns:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": content})

    payload = {"model": model_id, "messages": messages}
    if provider == "openai":
        # These models count reasoning tokens against the output budget, so
        # keep reasoning light for chat and give the answer room to finish.
        payload["max_completion_tokens"] = max(max_tokens, 4000)
        payload["reasoning_effort"] = "low"
    else:
        payload["max_tokens"] = max(max_tokens, 4000)

    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if r.status_code != 200:
        logger.error(f"{provider} returned {r.status_code}: {r.text[:400]}")
        raise HTTPException(status_code=502, detail=f"{TYPEMYWORDZ_AI_NAME} could not reach that model. Please try again.")
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    if choice.get("finish_reason") == "length" and text:
        text += "\n\n[The answer was cut short because it reached its length limit. Ask me to continue and I will pick up where I stopped.]"
    return text


def _ask_gemini(model_id, system_prompt, turns, question, images, max_tokens):
    """Call Gemini over HTTP and return the answer text."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail=f"{TYPEMYWORDZ_AI_NAME} is not connected to that model right now.")

    convo = [system_prompt]
    for turn in turns:
        convo.append(("You: " if turn["role"] == "user" else "Assistant: ") + turn["content"])
    convo.append("You: " + question)

    parts = [{"text": "\n\n".join(convo)}]
    for img in images:
        parts.append({"inline_data": {"mime_type": img["media_type"], "data": img["data"]}})

    gen = {"maxOutputTokens": max(max_tokens, GEMINI_MIN_OUTPUT_TOKENS)}
    if model_id in GEMINI_THINKING_OFF:
        gen["thinkingConfig"] = {"thinkingBudget": 0}

    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": parts}], "generationConfig": gen},
        timeout=180,
    )
    if r.status_code != 200:
        logger.error(f"gemini returned {r.status_code}: {r.text[:400]}")
        raise HTTPException(status_code=502, detail=f"{TYPEMYWORDZ_AI_NAME} could not reach that model. Please try again.")
    data = r.json()
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(
        p.get("text", "") for p in ((cand.get("content") or {}).get("parts") or [])
    ).strip()
    if cand.get("finishReason") == "MAX_TOKENS" and text:
        text += "\n\n[The answer was cut short because it reached its length limit. Ask me to continue and I will pick up where I stopped.]"
    if not text:
        raise HTTPException(status_code=502, detail=f"{TYPEMYWORDZ_AI_NAME} did not get an answer back. Please try again.")
    return text


def _ask_claude(model_id, system_prompt, turns, question, images, max_tokens):
    """Call Claude and return the answer text."""
    if not claude_client:
        raise HTTPException(status_code=503, detail=f"{TYPEMYWORDZ_AI_NAME} service is not initialized.")
    content = []
    for img in images:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": img["media_type"], "data": img["data"]},
        })
    content.append({"type": "text", "text": question})
    messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    messages.append({"role": "user", "content": content})
    response = claude_client.messages.create(
        model=model_id,
        max_tokens=max(max_tokens, 4000),
        system=system_prompt,
        messages=messages,
    )
    text = claude_text(response)
    if not text:
        # The model returned nothing usable. Better to say so than to show
        # the client an empty answer bubble and leave them guessing.
        text = (
            "That answer came back empty, which is a hiccup on the model's side "
            "rather than anything you did. Ask again, or pick a different model "
            "in Settings."
        )
    if getattr(response, "stop_reason", None) == "max_tokens" and text:
        text += "\n\n[The answer was cut short because it reached its length limit. Ask me to continue and I will pick up where I stopped.]"
    return text


@app.get("/credits/balance")
async def credits_balance(user_id: str = "", user_email: str = ""):
    """What this account can spend, and what each thing costs.

    The client shows this; it never decides it. Any refill or expiry noticed
    while reading is written back here, so the number a client sees is the
    number the server will actually honour.
    """
    if credits_exempt(user_email):
        return {"exempt": True, "unlimited": True, "planCredits": None,
                "topUpCredits": None, "total": None,
                "costs": {"transcription_per_minute": 1},
                "bundles": TOPUP_BUNDLES}

    if not user_id and user_email:
        user_id = await get_user_profile_by_email_firestore(user_email)
    profile = await _load_profile(user_id) if user_id else None
    if profile is None:
        raise HTTPException(status_code=404, detail="We could not find that account.")

    bal = read_balance(profile)
    if bal["updates"]:
        await _save_credit_updates(user_id, bal["updates"])

    return {
        "exempt": False,
        "unlimited": False,
        "planCredits": bal["planCredits"],
        "topUpCredits": bal["topUpCredits"],
        "total": bal["total"],
        "spendable": bal["spendable"],
        "frozen": bal["frozen"],
        "planActive": bal["planActive"],
        "planCreditsExpireAt": bal["planCreditsExpireAt"].isoformat() if bal["planCreditsExpireAt"] else None,
        "topUpCreditsExpireAt": bal["topUpCreditsExpireAt"].isoformat() if bal["topUpCreditsExpireAt"] else None,
        "costs": {
            "transcription_per_minute": 1,
            "ask": {m["id"]: m.get("credits", 1) for m in ASK_MODEL_CATALOGUE},
        },
        "bundles": TOPUP_BUNDLES,
    }


@app.get("/credits/quote")
async def credits_quote(seconds: float = 0, user_id: str = "", user_email: str = ""):
    """Can this account afford a file of this length?

    Asked before an upload starts, so that a client is told up front rather
    than after waiting for a transcript they cannot have.
    """
    cost = minutes_to_credits(seconds)
    if credits_exempt(user_email):
        return {"cost": 0, "affordable": True, "exempt": True, "balance": None}

    if not user_id and user_email:
        user_id = await get_user_profile_by_email_firestore(user_email)
    profile = await _load_profile(user_id) if user_id else None
    if profile is None:
        return {"cost": cost, "affordable": False, "exempt": False,
                "balance": 0, "short_by": cost}

    bal = read_balance(profile)
    return {
        "cost": cost,
        "balance": bal["total"],
        "spendable": bal["spendable"],
        "frozen": bal["frozen"],
        "planActive": bal["planActive"],
        "affordable": bal["spendable"] >= cost,
        "short_by": max(0, cost - bal["spendable"]),
        "exempt": False,
    }



@app.post("/human-transcription/quote")
async def human_transcription_quote(
    request: Request,
    seconds: float = Form(0),
    turnaround: str = Form("standard"),
    difficulty: str = Form("standard"),
    service: str = Form("standard"),
    speakers: str = Form("1-2"),
    timestamps: bool = Form(True),
    formatting: str = Form("standard"),
):
    """Quote a human transcription using the signed-in AI account.

    This endpoint is intentionally quote-only. It does not accept a browser
    supplied credit amount, upload a file, create a job, reserve credits, or
    deduct anything. Confirmation and idempotent reservation come later.
    """
    decoded = _verified_user(request)
    user_id = decoded.get("uid") or ""
    user_email = (decoded.get("email") or "").strip().lower()
    if not user_id or not user_email:
        raise HTTPException(status_code=401, detail="Your account could not be verified.")

    try:
        quote = human_credit_quote(seconds, turnaround, difficulty, service, speakers, timestamps, formatting)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if credits_exempt(user_email) or human_job_credits_exempt(user_email):
        return {
            **quote,
            "cost": 0,
            "spendable": None,
            "affordable": True,
            "exempt": True,
            "reservation": "not_created",
        }

    profile = await _load_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="We could not find that account.")

    balance = read_balance(profile)
    if balance["updates"]:
        await _save_credit_updates(user_id, balance["updates"])

    cost = quote["credits"]
    return {
        **quote,
        "cost": cost,
        "spendable": balance["spendable"],
        "affordable": balance["spendable"] >= cost,
        "short_by": max(0, cost - balance["spendable"]),
        "exempt": False,
        "reservation": "not_created",
    }


@app.post("/credits/backfill")
async def credits_backfill(user_id: str = Form(""), user_email: str = Form("")):
    """Convert one account from the old hours system to credits.

    Safe to call as often as the app likes: after the first time it does
    nothing. Clients who are mid-plan keep every minute they had left.
    """
    if credits_exempt(user_email):
        return {"exempt": True, "granted": 0}

    if not user_id and user_email:
        user_id = await get_user_profile_by_email_firestore(user_email)
    profile = await _load_profile(user_id) if user_id else None
    if profile is None:
        raise HTTPException(status_code=404, detail="We could not find that account.")

    updates, detail = backfill_credits(profile)
    if updates:
        await _save_credit_updates(user_id, updates, ledger_reason=detail.get("reason", "credit backfill"), ledger_context={"operation": "backfill"})
        logger.info(f"Credit backfill for {user_id}: {detail}")

    bal = read_balance({**profile, **updates})
    return {"exempt": False, "detail": detail,
            "planCredits": bal["planCredits"],
            "topUpCredits": bal["topUpCredits"],
            "total": bal["total"],
            "spendable": bal["spendable"],
            "frozen": bal["frozen"],
            "planActive": bal["planActive"]}


@app.get("/referrals/code")
async def referrals_code(user_id: str = "", user_email: str = ""):
    """This account's own referral code and running total, creating the code
    the first time it is asked for."""
    if not user_id and user_email:
        user_id = await get_user_profile_by_email_firestore(user_email)
    profile = await _load_profile(user_id) if user_id else None
    if profile is None:
        raise HTTPException(status_code=404, detail="We could not find that account.")

    code = await _ensure_referral_code(user_id, profile)
    completed = profile.get("referralsCompleted") or []
    return {
        "code": code,
        "share_url": f"https://typemywordz.ai/?ref={code}",
        "referral_bonus_credits": REFERRAL_BONUS_CREDITS,
        "referrals_completed": len(completed),
        "credits_earned": len(completed) * REFERRAL_BONUS_CREDITS,
    }


@app.post("/referrals/apply")
async def referrals_apply(user_id: str = Form(""), user_email: str = Form(""), code: str = Form("")):
    """A brand-new account redeems someone else's referral code.

    One-time per account, no self-referrals, and a bad or unknown code is
    reported quietly rather than as an error - a friend mistyping a code
    should never block someone from finishing signup.
    """
    code = (code or "").strip().upper()
    if not code:
        return {"applied": False, "reason": "no code given"}

    if not user_id and user_email:
        user_id = await get_user_profile_by_email_firestore(user_email)
    profile = await _load_profile(user_id) if user_id else None
    if profile is None:
        raise HTTPException(status_code=404, detail="We could not find that account.")

    if profile.get("referredByUid") or profile.get("referralAppliedAt"):
        return {"applied": False, "reason": "already applied"}

    code_snap = await asyncio.to_thread(db.collection("referral_codes").document(code).get)
    if not code_snap.exists:
        return {"applied": False, "reason": "unknown code"}
    referrer_uid = (code_snap.to_dict() or {}).get("uid")
    if not referrer_uid or referrer_uid == user_id:
        return {"applied": False, "reason": "invalid code"}

    referrer_profile = await _load_profile(referrer_uid)
    if referrer_profile is None:
        return {"applied": False, "reason": "referrer no longer exists"}

    now = datetime.now()

    referee_updates = _grant_referral_bonus(profile, now)
    referee_updates["referredByUid"] = referrer_uid
    referee_updates["referralAppliedAt"] = now
    await _save_credit_updates(user_id, referee_updates, ledger_reason="referral bonus", ledger_context={"role": "referee", "referrer_uid": referrer_uid})

    referrer_updates = _grant_referral_bonus(referrer_profile, now)
    referrer_updates["referralsCompleted"] = firestore.ArrayUnion([user_id])
    await _save_credit_updates(referrer_uid, referrer_updates, ledger_reason="referral bonus", ledger_context={"role": "referrer", "referee_uid": user_id})

    logger.info(f"Referral applied: {user_id} referred by {referrer_uid} via {code}")
    return {"applied": True, "bonus_credits": REFERRAL_BONUS_CREDITS}


@app.post("/paddle-custom-topup")
async def paddle_custom_topup(request: Request):
    """Create a Paddle checkout transaction for an exact global credit amount."""
    decoded = _verified_user(request)
    user_id = decoded.get("uid") or ""
    email = (decoded.get("email") or "").strip().lower()
    if not user_id or not email:
        raise HTTPException(status_code=401, detail="Your account could not be verified.")
    if not PADDLE_API_KEY or not PADDLE_CUSTOM_TOPUP_PRODUCT_ID:
        raise HTTPException(status_code=503, detail="International custom top-ups are not configured yet.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="The custom credit amount was not received.")
    try:
        credits = int(body.get("credits") or 0)
    except (TypeError, ValueError):
        credits = 0
    if custom_topup_credits(f"topup-custom-{credits}", "GLOBAL") is None:
        raise HTTPException(status_code=400, detail="Minimum allowed top up is 100 credits.")

    amount = custom_topup_price(credits, "GLOBAL")
    amount_minor = str(int(round(amount * 100)))
    item_id = f"topup-custom-{credits}"
    payload = {
        "items": [{
            "quantity": 1,
            "price": {
                "description": f"TypeMyworDz AI custom credit top-up: {credits} credits",
                "name": f"Custom top-up — {credits} credits",
                "product_id": PADDLE_CUSTOM_TOPUP_PRODUCT_ID,
                "unit_price": {"amount": amount_minor, "currency_code": "USD"},
            },
        }],
        "custom_data": {
            "user_id": user_id,
            "email": email,
            "item_id": item_id,
            "country_code": "GLOBAL",
        },
    }
    try:
        response = await asyncio.to_thread(
            requests.post,
            "https://api.paddle.com/transactions",
            headers={
                "Authorization": f"Bearer {PADDLE_API_KEY}",
                "Content-Type": "application/json",
                "Paddle-Version": "1",
            },
            json=payload,
            timeout=30,
        )
        data = response.json()
    except Exception as exc:
        logger.exception("Paddle custom top-up transaction creation failed")
        raise HTTPException(status_code=502, detail="Paddle checkout could not be reached.") from exc
    if not response.ok or not data.get("data"):
        logger.error("Paddle custom top-up rejected: %s", data)
        detail = (data.get("error") or {}).get("detail") if isinstance(data, dict) else None
        raise HTTPException(status_code=502, detail=detail or "Paddle could not create this checkout.")
    transaction = data["data"]
    return {
        "transaction_id": transaction.get("id"),
        "checkout_url": (transaction.get("checkout") or {}).get("url"),
        "credits": credits,
        "amount": amount,
        "currency": "USD",
    }


@app.get("/paddle-config")
async def paddle_config():
    """Return only the public Paddle checkout configuration.

    The client-side token and price IDs are safe to expose in the browser. The
    API key and webhook secret never leave the server.
    """
    if not PADDLE_CLIENT_TOKEN or not PADDLE_PRICE_IDS:
        raise HTTPException(status_code=503, detail="Paddle checkout is not configured yet.")
    return {
        "environment": "sandbox" if PADDLE_CLIENT_TOKEN.startswith("test_") else "live",
        "client_token": PADDLE_CLIENT_TOKEN,
        "price_ids": PADDLE_PRICE_IDS,
    }


def _paddle_signature_is_valid(raw_body: bytes, signature: str) -> bool:
    if not PADDLE_WEBHOOK_SECRET or not signature:
        return False
    parts = {}
    for piece in signature.split(";"):
        if "=" in piece:
            key, value = piece.split("=", 1)
            parts[key.strip()] = value.strip()
    timestamp = parts.get("ts")
    received = parts.get("h1")
    if not timestamp or not received:
        return False
    signed_payload = f"{timestamp}:".encode("utf-8") + raw_body
    expected = hmac.new(
        PADDLE_WEBHOOK_SECRET.encode("utf-8"), signed_payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, received)


@app.post("/paddle-webhook")
async def paddle_webhook(request: Request):
    """Fulfil a verified Paddle transaction exactly once per event."""
    raw_body = await request.body()
    signature = request.headers.get("Paddle-Signature", "")
    if not _paddle_signature_is_valid(raw_body, signature):
        raise HTTPException(status_code=401, detail="Invalid Paddle signature.")

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Invalid Paddle event body.")

    event_type = payload.get("event_type", "")
    if event_type not in {"transaction.completed", "transaction.paid"}:
        return {"status": "ignored", "event_type": event_type}

    data = payload.get("data") or {}
    event_id = payload.get("event_id") or payload.get("notification_id") or data.get("id")
    if not event_id:
        raise HTTPException(status_code=400, detail="Paddle event has no identifier.")

    # Paddle can retry a notification. A stored event makes retries harmless;
    # top-ups also carry their own last-reference guard as a second safety net.
    if db:
        existing = await asyncio.to_thread(db.collection("paddleEvents").document(str(event_id)).get)
        if existing.exists:
            return {"status": "already_processed", "event_id": event_id}

    items = data.get("items") or []
    first_item = items[0] if items else {}
    price = first_item.get("price") or {}
    price_id = price.get("id") or first_item.get("price_id")
    mapped_item_id = PADDLE_PRICE_TO_ITEM.get(str(price_id))

    custom_data = data.get("custom_data") or {}
    if isinstance(custom_data, str):
        try:
            custom_data = json.loads(custom_data)
        except json.JSONDecodeError:
            custom_data = {}
    custom_item_id = custom_data.get("item_id") if isinstance(custom_data, dict) else None
    if mapped_item_id == "topup-custom" or str(custom_item_id or "").startswith("topup-custom-"):
        quantity = first_item.get("quantity") or 1
        item_id = custom_item_id or f"topup-custom-{quantity}"
    else:
        item_id = mapped_item_id
    if not item_id:
        raise HTTPException(status_code=400, detail="Unknown Paddle price.")

    email = custom_data.get("email")
    customer = data.get("customer") or {}
    if not isinstance(customer, dict):
        customer = {}
    if not email:
        email = customer.get("email")
    if not email and custom_data.get("user_id"):
        try:
            email = (await asyncio.to_thread(firebase_auth.get_user, custom_data["user_id"])).email
        except Exception:
            email = None
    if not email:
        raise HTTPException(status_code=400, detail="Paddle transaction has no customer email.")

    totals = ((data.get("details") or {}).get("totals") or {})
    grand_total = totals.get("grand_total") or data.get("grand_total") or "0"
    try:
        amount = round(float(grand_total) / 100, 2)
    except (TypeError, ValueError):
        amount = 0.0
    currency = data.get("currency_code") or "USD"
    reference = data.get("id") or str(event_id)
    country_code = custom_data.get("country_code") or "GLOBAL"

    result = await update_user_credits_paystack(
        email=email,
        plan_name=item_id,
        amount=amount,
        currency=currency,
        update_admin_revenue=True,
        country_code=country_code,
        reference=reference,
    )
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Could not fulfil Paddle payment."))

    if db:
        await asyncio.to_thread(
            db.collection("paddleEvents").document(str(event_id)).set,
            {"eventType": event_type, "transactionId": reference, "email": email,
             "itemId": item_id, "processedAt": firestore.SERVER_TIMESTAMP},
        )
    return {"status": "processed", "event_id": event_id, "item_id": item_id}


@app.get("/pricing")
async def pricing(country_code: str = "GLOBAL"):
    """What we sell and what it costs, for one client.

    The page that shows this must not have to know that more than one price
    list exists. It asks, it is told one set of prices, and it displays them.
    """
    region = price_region(country_code)
    table = PRICES[region]

    plans = []
    for name in ['One-Day Plan', 'Three-Day Plan', 'One-Week Plan', 'Monthly Plan', 'Yearly Plan']:
        spec = PLAN_CREDITS.get(name)
        if not spec or name not in table:
            continue
        plans.append({
            'id': name,
            'price': table[name],
            'credits': spec['credits'],
            'days': spec['days'],
            'monthly_refill': spec['monthly_refill'],
            'premium_ai': name in PREMIUM_AI_PLANS,
        })

    topups = []
    for bundle_id in ['topup-300', 'topup-800', 'topup-2000']:
        if bundle_id in table:
            topups.append({
                'id': bundle_id,
                'price': table[bundle_id],
                'credits': TOPUP_BUNDLES[bundle_id],
            })

    return {
        'currency': 'USD',
        'plans': plans,
        'topups': topups,
        'custom_topup': {
            'min_credits': custom_topup_min(country_code),
            'max_credits': CUSTOM_TOPUP_MAX,
            'price_per_credit': CUSTOM_TOPUP_RATE[region],
        },
        'topup_valid_days': TOPUP_VALID_DAYS,
        'free_trial_credits': FREE_TRIAL_CREDITS,
    }


@app.post("/credits/topup")
async def credits_topup(
    bundle_id: str = Form(...),
    user_id: str = Form(""),
    user_email: str = Form(""),
    reference_id: str = Form(""),
):
    """Add a bought bundle of credits.

    The client sends only which bundle was bought. How many credits that is
    worth is decided here, from the server's own table, so that a client
    cannot ask for the small bundle and be given the large one.
    """
    if bundle_id not in TOPUP_BUNDLES and custom_topup_credits(bundle_id) is None:
        raise HTTPException(status_code=400, detail="That is not a top-up we sell.")

    # Credits are money. This endpoint will only add them against a payment
    # that Paystack itself confirms was made, for this exact bundle. Without
    # that check anyone who found the address could help themselves. An admin
    # may still add credits by hand, for support and goodwill.
    if not is_admin_user(user_email):
        if not reference_id:
            raise HTTPException(status_code=400, detail="A payment reference is required.")
        check = await verify_paystack_payment(reference_id)
        if check.get('status') != 'success':
            raise HTTPException(status_code=402, detail="That payment could not be confirmed.")
        if check.get('plan') != bundle_id:
            raise HTTPException(status_code=400, detail="That payment was not for this top-up.")
        if user_email and check.get('email') and check['email'].lower() != user_email.lower():
            raise HTTPException(status_code=403, detail="That payment belongs to another account.")

    if not user_id and user_email:
        user_id = await get_user_profile_by_email_firestore(user_email)
    profile = await _load_profile(user_id) if user_id else None
    if profile is None:
        raise HTTPException(status_code=404, detail="We could not find that account.")

    result = grant_topup_credits(profile, bundle_id)
    if not result:
        raise HTTPException(status_code=400, detail="That is not a top-up we sell.")

    updates = dict(result["updates"])
    if reference_id:
        updates["lastTopUpReference"] = reference_id
    updates["lastTopUpAt"] = firestore.SERVER_TIMESTAMP

    if not await _save_credit_updates(user_id, updates, ledger_reason="credit top-up purchase", ledger_context={"item": bundle_id, "payment_reference": reference_id}):
        raise HTTPException(status_code=500, detail="The credits could not be added. Please contact support.")

    logger.info(f"Top-up {bundle_id} ({result['added']} credits) added for {user_id}")
    return {"success": True, "added": result["added"], "total": result["newTotal"]}


@app.post("/ai/ask")
async def ai_ask(
    user_prompt: str = Form(...),
    history: str = Form(""),
    transcript: str = Form(""),
    # No default here on purpose. This used to default to "claude", which made
    # "the client did not choose a model" look identical to "the client asked
    # for Claude", so everyone silently got Claude Haiku instead of the
    # cheaper default the Settings page advertises.
    provider: str = Form(""),
    model: str = Form(""),
    max_tokens: int = Form(2000),
    user_plan: str = Form("free"),
    user_email: str = Form(""),
    user_id: str = Form(""),
    files: List[UploadFile] = File(default=[]),
):
    """Ask TypeMyworDz.

    One endpoint behind both places the assistant appears: the standalone page,
    and the panel beside a finished transcript. It keeps the conversation going
    via the history field, accepts attachments, and puts no limit on the length
    of the question.
    """
    logger.info(f"Ask endpoint called. model={model or provider}, plan={user_plan}, files={len(files or [])}")

    has_credits = await account_has_usable_credits(user_email=user_email)
    if not is_ai_allowed(user_plan, user_email, has_credits):
        raise HTTPException(status_code=403, detail="Ask TypeMyworDz is available on any paid plan. Choose a plan to switch it on.")

    if not (user_prompt or "").strip() and not files:
        raise HTTPException(status_code=400, detail="Please type a question.")

    # ---- attachments -----------------------------------------------------
    images = []
    doc_texts = []
    problems = []
    for uf in (files or [])[:MAX_ATTACHMENTS]:
        try:
            raw = await uf.read()
        except Exception:
            problems.append(f"{uf.filename}: it could not be uploaded")
            continue
        item = read_attachment(uf.filename, raw)
        if item['kind'] == 'image':
            images.append(item)
        elif item['kind'] == 'text':
            doc_texts.append(
                f"--- Attached file: {item['name']} ---\n{item['text']}"
                f"\n--- end of {item['name']} ---"
            )
        else:
            problems.append(f"{item['name']}: {item['message']}")

    # ---- build the question ----------------------------------------------
    pieces = []
    if transcript and transcript.strip():
        pieces.append(f"Here is the transcript being discussed:\n{transcript.strip()[:200000]}")
    if doc_texts:
        # Without this framing the text just looks like part of the question, and
        # weaker models reply "I cannot see an attached file" even though the
        # whole document is sitting right there in front of them. Measured: Claude
        # Haiku denied a PDF it had been given until this wrapper was added.
        count = len(doc_texts)
        pieces.append(
            f"The client attached {count} file{'' if count == 1 else 's'} to this "
            "message. The full contents are included below, so treat them as "
            "genuinely attached.\n\n" + "\n\n".join(doc_texts)
        )
    pieces.append((user_prompt or "Please look at what I have attached.").strip())
    question = "\n\n".join(pieces)

    turns = parse_history(history)

    # The client asks; the server decides. An id the plan does not include
    # falls back to the default rather than raising.
    has_transcript = bool(transcript and transcript.strip())
    ask_system_prompt = (
        ASK_SYSTEM_PROMPT_TRANSCRIPT if has_transcript else ASK_SYSTEM_PROMPT_GENERAL
    )
    chosen_model, chosen_provider = resolve_ask_model(
        model or provider, user_plan, user_email, has_transcript
    )

    try:
        if chosen_provider in OPENAI_FORMAT_ENDPOINTS:
            answer = _ask_openai_format(
                chosen_provider, chosen_model, ask_system_prompt, turns, question, images, max_tokens
            )
            model_used = chosen_model
        elif chosen_provider == "gemini":
            answer = _ask_gemini(
                chosen_model, ask_system_prompt, turns, question, images, max_tokens
            )
            model_used = chosen_model
        else:
            answer = _ask_claude(
                chosen_model, ask_system_prompt, turns, question, images, max_tokens
            )
            model_used = chosen_model

        cost = ask_credit_cost(chosen_model)
        usage_category = "transcript_query" if has_transcript else "standalone_ask"
        charge = await charge_credits(
            user_id or "", user_email, cost, f"ask {chosen_model}", usage_category=usage_category
        )

        return {
            "ai_response": answer,
            "model_used": model_used,
            "usage_category": usage_category,
            "credits_used": charge.get("charged", cost),
            "credits_remaining": charge.get("remaining"),
            "attachments_read": len(images) + len(doc_texts),
            "attachment_problems": problems,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")




@app.post("/ai/user-query-gemini")
async def ai_user_query_gemini(
    transcript: str = Form(...),
    user_prompt: str = Form(...),
    model: str = Form("models/gemini-pro-latest"),
    max_tokens: int = Form(1000),
    user_plan: str = Form("free"),
    user_email: str = Form("")
):
    """Same job as /ai/user-query, but answered by Gemini instead of Claude.

    This existed for the admin formatter but never for clients, so the
    frontend's Gemini option was calling a URL that returned 404.
    """
    logger.info(f"Gemini user query endpoint called. Model: {model}, User Plan: {user_plan}")

    has_credits = await account_has_usable_credits(user_email=user_email)
    if not is_ai_allowed(user_plan, user_email, has_credits):
        raise HTTPException(status_code=403, detail="AI Assistant features are only available on a paid plan. Please choose a plan to continue.")

    if not gemini_client:
        raise HTTPException(status_code=503, detail=f"{TYPEMYWORDZ_AI_NAME} Gemini service is not initialized (API key missing or invalid).")

    try:
        if len(transcript) > 200000:
            raise HTTPException(status_code=400, detail="Transcript is too long. Please use a shorter transcript.")

        full_prompt = f"{user_prompt}\n\nHere is the transcript:\n{transcript}"

        chosen = genai.GenerativeModel(model) if model and model != "models/gemini-pro-latest" else gemini_client
        response = chosen.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
        )
        ai_response = response.text
        logger.info(f"Successfully processed Gemini user query with {model}.")
        return {"ai_response": ai_response}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini user query failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


@app.post("/ai/admin-format")
async def ai_admin_format(
    transcript: str = Form(...),
    formatting_instructions: str = Form("Format the transcript for readability, correct grammar, and identify main sections with headings. Ensure a professional tone."),
    model: str = Form("claude-haiku-4-5-20251001"),
    max_tokens: int = Form(4000),
    user_plan: str = Form("free"),
    user_email: str = Form("")
):
    logger.info(f"AI admin format endpoint (Anthropic) called. Model: {model}, Instructions: '{formatting_instructions}', User Plan: {user_plan}")

    has_credits = await account_has_usable_credits(user_email=user_email)
    if not is_ai_allowed(user_plan, user_email, has_credits):
        raise HTTPException(status_code=403, detail="AI Admin formatting features are only available for paid AI users (Three-Day, One-Week, Monthly Plan, Yearly Plan plans). Please upgrade your plan.")

    if not claude_client:
        raise HTTPException(status_code=503, detail=f"{TYPEMYWORDZ_AI_NAME} service is not initialized (API key missing or invalid).")

    try:
        if len(transcript) > 200000:
            raise HTTPException(status_code=400, detail="Transcript is too long. Please use a shorter transcript.")
        
        full_prompt = f"Please apply the following formatting and polishing instructions to the provided transcript:\n\nInstructions: {formatting_instructions}\n\nTranscript to format:\n{transcript}"

        message = claude_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            timeout=60.0,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        ai_response = claude_text(message)
        logger.info(f"Successfully processed AI admin format request with {model}.")
        return {"formatted_transcript": ai_response}

    except anthropic.APIError as e:
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

@app.post("/ai/admin-format-gemini")
async def ai_admin_format_gemini(
    transcript: str = Form(...),
    formatting_instructions: str = Form("Correct all grammar, ensure a formal tone, break into paragraphs with subheadings for each major topic, and highlight action items in bold."),
    model: str = Form("models/gemini-pro-latest"),
    max_tokens: int = Form(4000),
    user_plan: str = Form("free"),
    user_email: str = Form("")
):
    logger.info(f"AI admin format endpoint (Gemini) called. Model: {model}, Instructions: '{formatting_instructions}', User Plan: {user_plan}")

    has_credits = await account_has_usable_credits(user_email=user_email)
    if not is_ai_allowed(user_plan, user_email, has_credits):
        raise HTTPException(status_code=403, detail="AI Admin formatting features are only available for paid AI users (Three-Day, One-Week, Monthly Plan, Yearly Plan plans). Please upgrade your plan.")

    if not gemini_client:
        logger.error("Gemini client is not initialized in /ai/admin-format-gemini. Check GEMINI_API_KEY.")
        raise HTTPException(status_code=503, detail=f"Google Gemini service is not initialized (API key missing or invalid).")

    try:
        if len(transcript) > 200000:
            raise HTTPException(status_code=400, detail="Transcript is too long. Please use a shorter transcript.")
        
        full_prompt = f"Please apply the following formatting and polishing instructions to the provided transcript:\n\nInstructions: {formatting_instructions}\n\nTranscript to format:\n{transcript}"

        response = gemini_client.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                top_k=40
            )
        )
        # Handle safety filter or empty response case
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            logger.warning(f"Gemini response was filtered or empty. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'}")
            return {"formatted_transcript": "The AI was unable to process this content due to content safety filters. Please try reformulating your request or use Claude instead."}
        
        gemini_response = response.text
        logger.info(f"Successfully processed AI admin format request with Gemini model: {model}.")
        return {"formatted_transcript": gemini_response}

    except Exception as e:
        logger.error(f"Unexpected error processing AI admin format request with Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during Gemini formatting: {str(e)}. Try using Claude instead.")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form("en"),
    speaker_labels_enabled: bool = Form(False),
    user_plan: str = Form("free"),
    user_email: str = Form(""),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    logger.info(f"Main transcription endpoint called with file: {file.filename}, language: {language_code}, speaker_labels: {speaker_labels_enabled}, user_plan: {user_plan}, user_email: {user_email}")
    
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
        
        jobs[job_id] = {
            "status": "processing",
            "filename": file.filename,
            "created_at": datetime.now().isoformat(),
            "file_size_mb": round(file_size_mb, 2),
            "content_type": file.content_type,
            "requested_language": language_code,
            "speaker_labels_enabled": speaker_labels_enabled,
            "user_plan": user_plan,
            "user_email": user_email,
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
        duration_minutes,
        user_email
    )
    
    logger.info(f"Returning immediate response for job ID: {job_id}")
    return {
        "job_id": job_id, 
        "status": jobs[job_id]["status"],
        "filename": file.filename,
        "file_size_mb": jobs[job_id]["file_size_mb"],
        "duration_minutes": duration_minutes,
        "created_at": jobs[job_id]["created_at"],
        "logic_used": f"UserPlan:{user_plan}, Email:{user_email}, Admin:{is_admin_user(user_email)}"
    }

@app.post("/generate-formatted-word")
async def generate_formatted_word(request: FormattedWordDownloadRequest):
    logger.info(f"Generating formatted Word document for {request.filename}")
    try:
        document = Document()
        lines = request.transcription_html.split('\n')
        
        speaker_tag_pattern = re.compile(r'<strong>(Speaker \d+:)</strong>(.*)')
        
        for line in lines:
            if line.strip():
                p = document.add_paragraph()
                
                match = speaker_tag_pattern.match(line)
                if match:
                    speaker_label_text = match.group(1)
                    remaining_text = match.group(2).strip()

                    run = p.add_run(speaker_label_text)
                    run.bold = True
                    
                    if remaining_text:
                        p.add_run(" " + remaining_text)
                else:
                    clean_line = re.sub(r'<[^>]*>', '', line).strip()
                    if clean_line:
                        p.add_run(clean_line)

        file_stream = BytesIO()
        document.save(file_stream)
        file_stream.seek(0)

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
        raise HTTPException(status_code=500, detail=f"Failed to compress audio file: {str(e)}")

@app.get("/jobs")
async def list_jobs():
    logger.info("Jobs list endpoint called")
    
    job_summary = {}
    for job_id, job_data in jobs.items():
        job_summary[job_id] = {
            "status": job_data["status"],
            "filename": job_data.get("filename", "unknown"),
            "created_at": datetime.fromisoformat(job_data["created_at"]).strftime('%Y-%m-%d %H:%M:%S'),
            "file_size_mb": job_data.get("file_size_mb", 0),
            "duration_minutes": job_data.get("duration_minutes", 0),
            "user_plan": job_data.get("user_plan", "unknown"),
            "user_email": job_data.get("user_email", ""),
            "is_admin": is_admin_user(job_data.get("user_email", "")),
            "primary_service": job_data.get("tier_1_service"),
            "service_used": (job_data.get("tier_1_used") or job_data.get("tier_2_used") or job_data.get("tier_3_used")),
            "model_used": job_data.get("model_used", "N/A"),
            "has_background_task": job_id in active_background_tasks,
            "is_cancellation_flagged": cancellation_flags.get(job_id, False),
            "word_count": job_data.get("word_count"),
            "duration_seconds": job_data.get("duration_seconds"),
            "requested_language": job_data.get("requested_language", "en"),
            "speaker_labels_enabled": job_data.get("speaker_labels_enabled", False),
            "selection_reason": job_data.get("selection_reason", "unknown")
        }
    
    return {
        "total_jobs": len(jobs),
        "active_background_tasks": len(active_background_tasks),
        "cancellation_flags": len(cancellation_flags),
        "jobs": job_summary,
        "admin_emails": ADMIN_EMAILS,
        "openai_tester_email": OPENAI_TESTER_EMAIL,
        "deepgram_tester_email": DEEPGRAM_TESTER_EMAIL,
        "gemini_access": "NOW AVAILABLE FOR ALL PAID AI USERS (Three-Day, One-Week, Monthly Plan, Yearly Plan plans)",
        "system_stats": {
            "jobs_by_status": {
                status: len([j for j in jobs.values() if j["status"] == status])
                for status in ["processing", "completed", "failed", "cancelled"]
            }
        }
    }

class WelcomeEmailRequest(BaseModel):
    email: str
    name: Optional[str] = ""


class AdminDeleteUserRequest(BaseModel):
    email: Optional[str] = None
    uid: Optional[str] = None


class FeedbackNotificationRequest(BaseModel):
    name: Optional[str] = ""
    email: str
    feedback: str


class TrafficEventRequest(BaseModel):
    visitorId: str
    page: str
    source: Optional[str] = "Direct"
    locale: Optional[str] = "unknown"
    timezone: Optional[str] = "unknown"


@app.post("/api/traffic-event")
async def traffic_event(payload: TrafficEventRequest):
    """Record anonymous page telemetry through the trusted backend."""
    if not db:
        return {"recorded": False, "reason": "not_configured"}
    event = {
        "visitorId": payload.visitorId[:120],
        "page": payload.page[:180],
        "source": (payload.source or "Direct")[:120],
        "locale": (payload.locale or "unknown")[:30],
        "timezone": (payload.timezone or "unknown")[:80],
        "createdAt": firestore.SERVER_TIMESTAMP,
    }
    await asyncio.to_thread(db.collection("trafficEvents").add, event)
    return {"recorded": True}


@app.get("/api/admin/traffic")
async def admin_traffic(request: Request):
    """Return recent traffic telemetry for the protected admin dashboard."""
    _require_admin(request)
    if not db:
        return {"events": []}
    cutoff = datetime.utcnow() - timedelta(days=30)
    events = []
    query = db.collection("trafficEvents").where(
        filter=FieldFilter("createdAt", ">=", cutoff)
    )
    for document in await asyncio.to_thread(lambda: list(query.stream())):
        data = document.to_dict() or {}
        created_at = data.get("createdAt")
        if hasattr(created_at, "isoformat"):
            data["createdAt"] = created_at.isoformat()
        data["id"] = document.id
        events.append(data)
    return {"events": events}


@app.get("/api/admin/users")
async def admin_users(request: Request):
    """Return the admin account snapshot from Firebase Admin SDK."""
    _require_admin(request)
    if not db:
        raise HTTPException(status_code=503, detail="The admin data service is not available.")
    return {"users": await asyncio.to_thread(_read_admin_users_snapshot)}


@app.post("/api/admin/delete-user")
async def admin_delete_user(payload: AdminDeleteUserRequest, request: Request):
    """Delete an account and its owned app data from the admin dashboard.

    Most accounts are identified by email, but some Firestore profiles were
    created without one ever being saved (an incomplete signup, for example).
    Those still carry a uid, so accept uid-only requests and look the email up
    from Firebase Auth when we have it, instead of hard-requiring email.
    """
    _require_admin(request)
    email = (payload.email or "").strip().lower()
    auth_uid = (payload.uid or "").strip()

    if not email and not auth_uid:
        raise HTTPException(status_code=400, detail="An email address or account ID is required.")

    if auth_uid and not email:
        try:
            record = await asyncio.to_thread(firebase_auth.get_user, auth_uid)
            email = (record.email or "").strip().lower()
        except firebase_auth.UserNotFoundError:
            email = ""
        except Exception as exc:
            logger.error("Could not look up auth account %s: %s", auth_uid, exc)

    if email and email in {item.lower() for item in ADMIN_EMAILS}:
        raise HTTPException(status_code=400, detail="Admin accounts cannot be deleted here.")

    if not auth_uid and email:
        try:
            record = await asyncio.to_thread(firebase_auth.get_user_by_email, email)
            auth_uid = record.uid
        except firebase_auth.UserNotFoundError:
            auth_uid = ""
        except Exception as exc:
            logger.error("Could not look up auth account %s: %s", email, exc)

    deleted = {"auth": False, "profiles": 0, "transcriptions": 0, "chats": 0, "feedback": 0}
    if auth_uid:
        try:
            await asyncio.to_thread(firebase_auth.delete_user, auth_uid)
            deleted["auth"] = True
        except firebase_auth.UserNotFoundError:
            pass

    if db:
        if auth_uid:
            await asyncio.to_thread(db.collection("users").document(auth_uid).delete)
            deleted["profiles"] += 1
            deleted["transcriptions"] = await asyncio.to_thread(
                _delete_matching_documents, "transcriptions", "userId", auth_uid
            )
            deleted["chats"] = await asyncio.to_thread(
                _delete_matching_documents, "askChats", "userId", auth_uid
            )
        if email:
            deleted["profiles"] += await asyncio.to_thread(
                _delete_matching_documents, "users", "email", email
            )
            deleted["feedback"] = await asyncio.to_thread(
                _delete_matching_documents, "feedback", "email", email
            )

    logger.warning("Admin deleted account %s (uid=%s): %s", email or "(no email)", auth_uid or "(no uid)", deleted)
    return {"success": True, "email": email, "uid": auth_uid, "deleted": deleted}


@app.post("/api/feedback-notification")
async def feedback_notification(payload: FeedbackNotificationRequest, request: Request):
    """Email the support mailbox after authenticated in-app feedback is saved."""
    decoded = _verified_user(request)
    signed_in_email = (decoded.get("email") or "").strip().lower()
    sender_email = (payload.email or "").strip().lower()
    if not sender_email or sender_email != signed_in_email:
        raise HTTPException(status_code=403, detail="Feedback email does not match the signed-in account.")
    if not payload.feedback.strip():
        raise HTTPException(status_code=400, detail="Feedback cannot be empty.")
    if not RESEND_API_KEY:
        return {"sent": False, "reason": "not_configured"}

    clean_name = escape(payload.name or "Anonymous")
    clean_email = escape(sender_email)
    clean_feedback = escape(payload.feedback.strip()).replace("\n", "<br>")
    subject = "New TypeMyworDz feedback from %s" % sender_email
    html = (
        "<div style=\"font-family:Arial,sans-serif;color:#14161a;max-width:640px\">"
        "<h2>New TypeMyworDz feedback</h2>"
        "<p><b>From:</b> %s &lt;%s&gt;</p><p style=\"white-space:pre-wrap\">%s</p>"
        "</div>"
    ) % (clean_name, clean_email, clean_feedback)
    text = "New TypeMyworDz feedback\n\nFrom: %s <%s>\n\n%s" % (payload.name or "Anonymous", sender_email, payload.feedback.strip())
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                RESEND_ENDPOINT,
                headers={"Authorization": "Bearer %s" % RESEND_API_KEY, "Content-Type": "application/json"},
                json={
                    "from": EMAIL_FROM,
                    "to": [SUPPORT_EMAIL],
                    "reply_to": sender_email,
                    "subject": subject,
                    "html": html,
                    "text": text,
                },
            )
        if response.status_code >= 400:
            logger.error("Feedback notification rejected: %s %s", response.status_code, response.text[:300])
            return {"sent": False, "reason": "provider_error"}
        return {"sent": True}
    except Exception as exc:
        logger.error("Feedback notification failed: %s", exc)
        return {"sent": False, "reason": "exception"}


async def _send_resend_message(address: str, subject: str, html: str, text: str, label: str = "email"):
    """Send a transactional message without making the calling workflow fail."""
    address = (address or "").strip()
    if not address or "@" not in address:
        return {"sent": False, "reason": "invalid_address"}
    if not RESEND_API_KEY:
        logger.warning("%s skipped for %s: RESEND_API_KEY is not set", label, address)
        return {"sent": False, "reason": "not_configured"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                RESEND_ENDPOINT,
                headers={
                    "Authorization": "Bearer %s" % RESEND_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "from": EMAIL_FROM,
                    "to": [address],
                    "reply_to": SUPPORT_EMAIL,
                    "subject": subject,
                    "html": html,
                    "text": text,
                },
            )
        if response.status_code >= 400:
            logger.error("%s rejected for %s: %s %s", label, address, response.status_code, response.text[:300])
            return {"sent": False, "reason": "provider_error"}
        logger.info("%s sent to %s", label, address)
        return {"sent": True}
    except Exception as exc:
        logger.error("%s failed for %s: %s", label, address, exc)
        return {"sent": False, "reason": "exception"}


def build_trainee_welcome_email(name: str):
    """Build the paid trainee welcome message without mentioning client credits."""
    raw_name = (name or "").strip()
    first = escape(raw_name.split(" ")[0]) if raw_name else "there"
    greeting = "Welcome, %s" % first
    subject = "Your TypeMyworDz training account is ready"
    html = """<!doctype html>
<html>
  <body style="margin:0;padding:0;background:#f8f8f9;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f8f8f9;padding:32px 16px;">
      <tr><td align="center">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:560px;background:#ffffff;border:1px solid #e5e6ea;border-radius:10px;padding:32px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
          <tr><td style="font-size:20px;font-weight:700;color:#14161a;padding-bottom:4px;">
            <span style="color:#5b44cf;">Type</span><span style="color:#28a745;">My</span><span style="color:#5b44cf;">worDz</span>
          </td></tr>
          <tr><td style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#858a95;padding-bottom:24px;">Your everyday AI companion</td></tr>
          <tr><td style="font-size:22px;font-weight:700;color:#14161a;padding-bottom:12px;">GREETING</td></tr>
          <tr><td style="font-size:15px;line-height:1.6;color:#3f434c;padding-bottom:16px;">Your paid training enrolment is confirmed and your Training Room account is ready.</td></tr>
          <tr><td style="font-size:15px;line-height:1.6;color:#3f434c;padding-bottom:16px;">Sign in to review the TypeMyworDz standards, work through the training modules, and submit each exercise for review. Your progress is saved in the Training Room.</td></tr>
          <tr><td style="font-size:15px;line-height:1.6;color:#3f434c;padding-bottom:24px;">Training is a skills programme and does not guarantee employment. If you need help, reply to this email or contact SUPPORT.</td></tr>
          <tr><td style="padding-bottom:28px;"><a href="APPURL" style="display:inline-block;background:#28a745;color:#ffffff;text-decoration:none;font-size:15px;font-weight:600;padding:12px 22px;border-radius:7px;">Open Training Room</a></td></tr>
          <tr><td style="font-size:12px;color:#858a95;padding-top:20px;border-top:1px solid #e5e6ea;">You are receiving this because a TypeMyworDz training account was created with this address.</td></tr>
        </table>
      </td></tr>
    </table>
  </body>
</html>"""
    html = html.replace("GREETING", greeting).replace("SUPPORT", SUPPORT_EMAIL).replace("APPURL", APP_URL)
    text = (
        "%s\n\n"
        "Your paid training enrolment is confirmed and your TypeMyworDz Training Room account is ready.\n\n"
        "Sign in here: %s\n\n"
        "Training is a skills programme and does not guarantee employment. For help, contact %s.\n"
    ) % ("Welcome, %s" % (raw_name.split(" ")[0] if raw_name else "there"), APP_URL, SUPPORT_EMAIL)
    return subject, html, text


def build_welcome_email(name: str, free_credits: int = None):
    """Build the subject and HTML body of the welcome email.

    Kept as a plain function with no network access so it can be unit tested.
    House style: no emoji, green only for things the client can act on,
    neutral greys for everything else.
    """
    if free_credits is None:
        free_credits = FREE_TRIAL_CREDITS
    first = (name or "").strip().split(" ")[0] if (name or "").strip() else ""
    greeting = "Welcome, %s" % first if first else "Welcome to TypeMyworDz"
    subject = "Welcome to TypeMyworDz AI"

    html = """<!doctype html>
<html>
  <body style="margin:0;padding:0;background:#f8f8f9;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f8f8f9;padding:32px 16px;">
      <tr>
        <td align="center">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
                 style="max-width:560px;background:#ffffff;border:1px solid #e5e6ea;border-radius:10px;padding:32px;
                        font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
            <tr><td style="font-size:20px;font-weight:700;color:#14161a;padding-bottom:4px;">
              <span style="color:#5b44cf;">Type</span><span style="color:#28a745;">My</span><span style="color:#5b44cf;">worDz</span>
            </td></tr>
            <tr><td style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#858a95;padding-bottom:24px;">
              Your everyday AI companion
            </td></tr>
            <tr><td style="font-size:22px;font-weight:700;color:#14161a;padding-bottom:12px;">GREETING</td></tr>
            <tr><td style="font-size:15px;line-height:1.6;color:#3f434c;padding-bottom:16px;">
              Your account is ready, with CREDITS free credits on it. One credit is one minute of
              audio, so that is CREDITS minutes of transcription to judge the quality on your own
              recordings before you decide anything.
            </td></tr>
            <tr><td style="font-size:15px;line-height:1.6;color:#3f434c;padding-bottom:16px;">
              Upload a recording, and when it is done you can proofread it against the audio,
              add speaker labels, copy it in one click, or export it as a Word or text file.
            </td></tr>
            <tr><td style="font-size:15px;line-height:1.6;color:#3f434c;padding-bottom:24px;">
              The same credits also cover Ask TypeMyworDz, which answers questions about your
              transcripts and anything else you need writing, researching or tidying up. You are
              never paying twice for the two halves.
            </td></tr>
            <tr><td style="padding-bottom:28px;">
              <a href="APPURL" style="display:inline-block;background:#28a745;color:#ffffff;text-decoration:none;
                 font-size:15px;font-weight:600;padding:12px 22px;border-radius:7px;">Start transcribing</a>
            </td></tr>
            <tr><td style="font-size:14px;line-height:1.6;color:#3f434c;border-top:1px solid #e5e6ea;padding-top:20px;">
              One thing worth knowing: we keep your transcripts, but we delete the audio as soon as it has been
              transcribed. Keep your own copy of any recording you may want to proofread against later.
            </td></tr>
            <tr><td style="font-size:14px;line-height:1.6;color:#3f434c;padding-top:16px;">
              Any questions, just reply to this address or write to
              <a href="mailto:SUPPORT" style="color:#28a745;">SUPPORT</a>. A real person answers.
            </td></tr>
            <tr><td style="font-size:12px;color:#858a95;padding-top:24px;">
              You are receiving this because an account was created with this address at TypeMyworDz.
            </td></tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>"""

    html = html.replace("GREETING", greeting)
    html = html.replace("CREDITS", str(free_credits))
    html = html.replace("APPURL", APP_URL)
    html = html.replace("SUPPORT", SUPPORT_EMAIL)

    text = (
        "%s\n\n"
        "Your account is ready, with %d free credits on it. One credit is one minute of audio.\n\n"
        "The same credits also cover Ask TypeMyworDz, which answers questions about your "
        "transcripts and anything else you need writing, researching or tidying up.\n\n"
        "Start here: %s\n\n"
        "One thing worth knowing: we keep your transcripts, but we delete the audio as soon as it "
        "has been transcribed. Keep your own copy of any recording you may want to proofread against later.\n\n"
        "Any questions, write to %s. A real person answers.\n"
    ) % (greeting, free_credits, APP_URL, SUPPORT_EMAIL)

    return subject, html, text


@app.post("/api/send-welcome-email")
async def send_welcome_email(payload: WelcomeEmailRequest):
    """Send the one-off welcome email to a brand new account.

    This deliberately never fails loudly. Signing up must not break because an
    email provider is having a bad day, so every problem is logged and reported
    back as sent=false instead of raising.
    """
    subject, html, text = build_welcome_email(payload.name or "")
    return await _send_resend_message(payload.email, subject, html, text, "Welcome email")


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
                "anthropic_configured": bool(ANTHROPIC_API_KEY),
                "openai_configured": bool(OPENAI_API_KEY),
                "openai_whisper_service_configured": bool(OPENAI_WHISPER_SERVICE_RAILWAY_URL),
                "google_gemini_configured": bool(GEMINI_API_KEY),
                "deepgram_service_configured": bool(DEEPGRAM_SERVICE_RAILWAY_URL)
            },
            "transcription_logic": {
                "free_user_transcription": f"Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
                "three_day_plan_transcription": f"Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
                "one_week_plan_transcription": f"Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
                "monthly_plan_transcription": f"Primary={TYPEMYWORDZ2_NAME} → Fallback1={TYPEMYWORDZ1_NAME} → Fallback2={DEEPGRAM_NAME}",
                "yearly_plan_transcription": f"Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
                "admin_transcription": f"Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
                "speaker_labels_transcription": f"Always use {TYPEMYWORDZ1_NAME} first → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}",
                "openai_tester_transcription": f"Always use {TYPEMYWORDZ2_NAME} (no fallback for {OPENAI_TESTER_EMAIL})",
                "deepgram_tester_transcription": f"Primary=Deepgram → Fallback=OpenAI for {DEEPGRAM_TESTER_EMAIL}",
                "assemblyai_models": f"{TYPEMYWORDZ1_NAME} universal-3-5-pro, falling back to universal-2 for other languages",
                "ai_features_access": "Only for Three-Day, One-Week, Monthly Plan, and Yearly Plan plans",
                "gemini_access": "NOW AVAILABLE FOR ALL PAID AI USERS (Three-Day, One-Week, Monthly Plan, Yearly Plan plans)",
                "assemblyai": f"TypeMyworDz1 (AssemblyAI)",
                "openai_whisper": f"TypeMyworDz2 (OpenAI Whisper-1)",
                "deepgram": f"Deepgram",
                "anthropic_ai": f"TypeMyworDz AI (Anthropic Claude)",
                "google_gemini_ai": "Google Gemini - Available for ALL paid AI users",
                "admin_emails": ADMIN_EMAILS,
                "openai_tester_email": OPENAI_TESTER_EMAIL,
                "deepgram_tester_email": DEEPGRAM_TESTER_EMAIL
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
logger.info(f"TypeMyworDz1 API Key configured: {bool(ASSEMBLYAI_API_KEY)}")
logger.info(f"TypeMyworDz2 Service URL configured: {bool(OPENAI_WHISPER_SERVICE_RAILWAY_URL)}")
logger.info(f"Deepgram Service URL configured: {bool(DEEPGRAM_SERVICE_RAILWAY_URL)}")
logger.info(f"TypeMyworDz AI API Key configured: {bool(ANTHROPIC_API_KEY)}")
logger.info(f"OpenAI GPT API Key configured: {bool(OPENAI_API_KEY)}")
logger.info(f"Google Gemini API Key configured: {bool(GEMINI_API_KEY)}")
logger.info(f"Paystack Secret Key configured: {bool(PAYSTACK_SECRET_KEY)}")
logger.info(f"Firebase Admin SDK configured: {bool(FIREBASE_ADMIN_SDK_CONFIG_BASE64) and bool(db)}")
logger.info(f"Admin emails configured: {ADMIN_EMAILS}")
logger.info(f"OpenAI Tester email configured: {OPENAI_TESTER_EMAIL}")
logger.info(f"UPDATED: Google Gemini now available for ALL PAID AI USERS (Three-Day, One-Week, Monthly Plan, Yearly Plan plans)")
logger.info(f"Job tracking systems initialized:")
logger.info(f"  - Main jobs dictionary: {len(jobs)} jobs")
logger.info(f"  - Active background tasks: {len(active_background_tasks)} tasks")
logger.info(f"  - Cancellation flags: {len(cancellation_flags)} flags")
logger.info("Available API endpoints:")
logger.info("  POST /transcribe - Main transcription endpoint with smart service selection")
logger.info("  POST /ai/user-query - Process user-driven AI queries (summarize, Q&A, bullet points) with Claude")
logger.info("  POST /ai/user-query-gemini - Process user-driven AI queries with Gemini (NOW FOR ALL PAID USERS)")
logger.info("  POST /ai/ask - Ask TypeMyworDz: conversation with attachments, no length limit")
logger.info("  POST /ai/admin-format - Process admin-driven AI formatting requests (Anthropic)")
logger.info("  POST /ai/admin-format-gemini - Process admin-driven AI formatting requests (Google Gemini - NOW FOR ALL PAID USERS)")
logger.info("  POST /api/initialize-paystack-payment - Initialize Paystack payment")
logger.info("  POST /api/verify-payment - Verify Paystack payment")
logger.info("  POST /api/paystack-webhook - Handle Paystack webhooks")
logger.info("  GET /api/paystack-status - Get integration status")
logger.info("  GET /api/list-gemini-models - List available Gemini models")
logger.info("  GET /status/{job_id} - Check job status")
logger.info("  POST /cancel/{job_id} - Cancel transcription job")
logger.info("  POST /compress-download - Compress audio for download")
logger.info("  POST /generate-formatted-word - Generate formatted Word document with speaker labels")
logger.info("  GET /jobs - List all jobs")
logger.info("  GET /health - System health check")
logger.info("  DELETE /cleanup - Clean up old jobs")
logger.info("  GET / - Root endpoint with service info")


@app.get("/credits/ledger")
async def credits_ledger(request: Request, limit: int = 50, user_id: str = "", email: str = ""):
    """Return the signed-in client's auditable credit history."""
    decoded = _verified_user(request)
    actor_uid = decoded.get("uid") or ""
    actor_email = (decoded.get("email") or "").strip().lower()
    if is_admin_user(actor_email) and (user_id or email):
        target_uid = user_id or await get_user_profile_by_email_firestore(email.strip().lower())
    else:
        target_uid = actor_uid
    if not target_uid or not db:
        return {"entries": []}
    try:
        snapshots = await asyncio.to_thread(lambda: list(db.collection("users").document(target_uid).collection("credit_ledger").stream()))
    except Exception as exc:
        logger.warning("Could not read credit ledger for %s: %s", target_uid, exc)
        return {"entries": []}
    entries = []
    for snap in snapshots:
        item = snap.to_dict() or {}
        item["id"] = snap.id
        for key, value in list(item.items()):
            if hasattr(value, "isoformat"):
                item[key] = value.isoformat()
        entries.append(item)
    entries.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return {"entries": entries[:max(1, min(int(limit or 50), 100))]}


@app.post("/api/admin/credits/adjust")
async def admin_adjust_credits(payload: AdminCreditAdjustmentRequest, request: Request):
    """Add or remove credits with an explicit admin reason and audit entry."""
    admin = _require_admin(request)
    if not db or not int(payload.amount):
        raise HTTPException(status_code=400, detail="Enter a non-zero credit adjustment.")
    target_uid = str(payload.user_id or "").strip()
    target_email = str(payload.email or "").strip().lower()
    if not target_uid and target_email:
        target_uid = await get_user_profile_by_email_firestore(target_email)
    if not target_uid:
        raise HTTPException(status_code=404, detail="The client account could not be found.")
    profile = await _load_profile(target_uid)
    if profile is None:
        raise HTTPException(status_code=404, detail="The client account could not be found.")
    delta = int(payload.amount)
    context = dict(payload.context or {})
    context.update({"admin_email": (admin.get("email") or "").lower(), "operation": "admin_adjustment"})
    if delta > 0:
        balance = read_balance(profile)
        updates = dict(balance.get("updates") or {})
        updates["topUpCredits"] = balance["topUpCredits"] + delta
        updates["topUpCreditsExpireAt"] = datetime.now() + timedelta(days=TOPUP_VALID_DAYS)
        if not await _save_credit_updates(target_uid, updates, ledger_reason=payload.reason, ledger_context=context):
            raise HTTPException(status_code=500, detail="The credit adjustment could not be saved.")
    else:
        ok, updates, detail = plan_spend(profile, abs(delta))
        if not ok:
            raise HTTPException(status_code=409, detail="The account does not have enough spendable credits to remove that amount.")
        if not await _save_credit_updates(target_uid, updates, ledger_reason=payload.reason, ledger_context=context):
            raise HTTPException(status_code=500, detail="The credit adjustment could not be saved.")
    refreshed = await _load_profile(target_uid) or {}
    current = read_balance(refreshed)
    return {"success": True, "user_id": target_uid, "amount": delta, "reason": payload.reason, "balance": current["spendable"]}


# ===================== Human workflow =====================
# The human service is deliberately a state machine.  A quote never charges
# credits; the client cannot download the finished work until the admin has
# released it after the client's approval.
HUMAN_JOB_COLLECTION = "human_jobs"
HUMAN_JOB_STATUSES = {
    "pending_admin",
    "approved",
    "assigned",
    "in_progress",
    "split_assigned",
    "split_in_progress",
    "proofreading_available",
    "proofreading_assigned",
    "proofreading_in_progress",
    "submitted",
    "client_review",
    "client_approved",
    "released",
    "cancelled",
}


def _human_iso(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _human_public(data):
    out = dict(data or {})
    for key, value in list(out.items()):
        if isinstance(value, dict):
            out[key] = _human_public(value)
        elif isinstance(value, list):
            out[key] = [_human_public(item) if isinstance(item, dict) else _human_iso(item) for item in value]
        else:
            out[key] = _human_iso(value)
    return out


# Workers must never see who the client is, and clients must never see who
# the worker is: every human job goes through admin, and admin is the only
# party allowed to see both sides. This strips the other side's identity off
# a job dict before it goes out, on top of the ordinary timestamp cleanup.
_HUMAN_CLIENT_IDENTITY_FIELDS = ("client_uid", "client_email", "client_name")
_HUMAN_WORKER_IDENTITY_FIELDS = ("worker_uid", "worker_email", "worker_name")


def _human_worker_segment(segments, actor_uid):
    """Select this worker's active part first, then their most recent part.

    A worker may be assigned a later split part after finishing an earlier
    one. Choosing the first matching segment would keep exposing the submitted
    part and hide the new active assignment.
    """
    owned = [item for item in (segments or []) if (item or {}).get("worker_uid") == actor_uid]
    active = next((item for item in owned if item.get("status") in {"assigned", "in_progress"}), None)
    return active or (owned[-1] if owned else None)


def _human_public_for(data, actor_role, actor_uid=""):
    """Serialize a human job without leaking the other side's identity.

    Split jobs keep one parent record but expose only the current worker's
    segment to that worker. Admins see the full assignment map.
    """
    data = data or {}
    out = _human_public(data)
    segments = data.get("segments") or []
    if actor_role in {"worker", "client"}:
        for key in (
            "last_assignment_takeback_at", "last_assignment_takeback_worker_uid",
            "last_assignment_takeback_worker_name", "last_assignment_takeback_role",
            "last_assignment_takeback_label", "last_assignment_takeback_by_uid",
            "last_assignment_takeback_by_email",
        ):
            out.pop(key, None)

    def remaining(deadline, status):
        deadline_dt = _as_dt(deadline)
        if deadline_dt is not None and status in ("assigned", "in_progress"):
            return max(0, int((deadline_dt - datetime.now()).total_seconds()))
        return None

    if actor_role == "worker":
        out.pop("last_message", None)
        out.pop("last_message_by_thread", None)
        for key in _HUMAN_CLIENT_IDENTITY_FIELDS:
            out.pop(key, None)
        for key in ("quote_credits", "credits_charged", "credits_deducted", "credit_cost", "price", "amount", "currency", "quote"):
            out.pop(key, None)
        owned_segment = _human_worker_segment(segments, actor_uid)
        assignment = dict(owned_segment) if owned_segment else None
        if assignment is not None:
            assignment["time_remaining_seconds"] = remaining(assignment.get("deadlineAt"), assignment.get("status"))
            assignment["role"] = "transcriber"
        if assignment is None and data.get("worker_uid") == actor_uid:
            assignment = {
                "id": "transcriber", "label": "Full transcript", "role": "transcriber",
                "status": data.get("status"), "transcript": data.get("transcript") or "",
                "final_attachment": _human_public(data.get("final_attachment") or {}) if data.get("final_attachment") else None,
                "assignedAt": data.get("assignedAt"), "deadlineAt": data.get("deadlineAt"),
                "time_remaining_seconds": remaining(data.get("deadlineAt"), data.get("status")),
                "start_seconds": 0, "end_seconds": data.get("seconds"),
            }
        if data.get("proofreader_uid") == actor_uid:
            assignment = {
                "id": "proofreader",
                "label": "Final proofreading",
                "role": "proofreader",
                "status": data.get("proofreader_status"),
                "transcript": data.get("transcript") or "\n\n".join(
                    item.get("transcript", "") for item in segments if item.get("transcript")
                ),
                "assignedAt": data.get("proofreader_assignedAt"),
                # The proofreader must start with a clean final-file input;
                # the earlier segment attachments are presented separately.
                "final_attachment": None,
                "deadlineAt": data.get("proofreader_deadlineAt"),
                "time_remaining_seconds": remaining(data.get("proofreader_deadlineAt"), data.get("proofreader_status")),
                "worker_minutes": data.get("minutes"),
            }
            out["proofreader_parts"] = [
                {
                    "id": item.get("id"),
                    "label": item.get("label") or f"Part {index + 1}",
                    "status": item.get("status"),
                    "transcript": item.get("transcript") or "",
                    "final_attachment": _human_public(item.get("final_attachment") or {}) if item.get("final_attachment") else None,
                }
                for index, item in enumerate(segments)
                if item.get("status") == "submitted"
            ]
        if assignment is not None:
            for key in ("worker_amount_kes", "worker_gross_amount_kes", "worker_deduction_kes", "worker_deduction_reason", "workerPaymentStatus", "workerPaidAt", "payout_status", "payout_period_id"):
                assignment.pop(key, None)
            assignment = _human_public(assignment)
            out["worker_assignment"] = assignment
            out["time_remaining_seconds"] = assignment.get("time_remaining_seconds")
            out["transcript"] = assignment.get("transcript") or ""
            out["final_attachment"] = assignment.get("final_attachment")
        for key in (
            "worker_minutes", "worker_amount_kes", "worker_gross_amount_kes", "worker_deduction_kes", "worker_deduction_reason",
            "workerPaymentStatus", "workerPaidAt", "payout_status", "payout_period_id",
            "proofreader_minutes", "proofreader_amount_kes", "proofreader_gross_amount_kes", "proofreader_deduction_kes", "proofreader_deduction_reason",
            "proofreader_payout_status", "proofreader_payout_period_id", "proofreaderPaidAt",
        ):
            out.pop(key, None)
        out.pop("segments", None)
        out.pop("assigned_worker_uids", None)
        out.pop("proofreader_uid", None)
        out.pop("proofreader_email", None)
        out.pop("proofreader_name", None)
        out.pop("proofreader_status", None)
        out.pop("proofreader_deadlineAt", None)
        out.pop("proofreader_tat_seconds", None)
        out.pop("proofreader_completedAt", None)
        out.pop("proofreader_minutes", None)
        out.pop("proofreader_amount_kes", None)
        out.pop("proofreader_payout_status", None)
        out.pop("proofreader_payout_period_id", None)
    elif actor_role == "client":
        out.pop("last_message", None)
        out.pop("last_message_by_thread", None)
        for key in _HUMAN_WORKER_IDENTITY_FIELDS:
            out.pop(key, None)
        for segment in out.get("segments") or []:
            if isinstance(segment, dict):
                for key in _HUMAN_WORKER_IDENTITY_FIELDS:
                    segment.pop(key, None)
        out.pop("assigned_worker_uids", None)
        out.pop("segments", None)
        for key in ("proofreader_uid", "proofreader_email", "proofreader_name"):
            out.pop(key, None)
    else:
        # Human-work admins get assignment details and useful countdowns.
        public_segments = []
        for item in out.get("segments") or []:
            if isinstance(item, dict):
                item["time_remaining_seconds"] = remaining(item.get("deadlineAt"), item.get("status"))
                if actor_role == "human_ops_admin":
                    for key in ("worker_minutes", "worker_amount_kes", "worker_gross_amount_kes", "worker_deduction_kes", "worker_deduction_reason", "workerPaymentStatus", "workerPaidAt", "payout_status", "payout_period_id"):
                        item.pop(key, None)
            public_segments.append(item)
        out["segments"] = public_segments
        out["time_remaining_seconds"] = remaining(data.get("deadlineAt"), data.get("status"))
        if data.get("proofreader_status") in ("assigned", "in_progress"):
            out["proofreader_time_remaining_seconds"] = remaining(data.get("proofreader_deadlineAt"), data.get("proofreader_status"))
        if actor_role == "human_ops_admin":
            for key in (
                "worker_minutes", "worker_amount_kes", "worker_gross_amount_kes", "worker_deduction_kes", "worker_deduction_reason",
                "workerPaymentStatus", "workerPaidAt", "payout_status", "payout_period_id",
                "proofreader_minutes", "proofreader_amount_kes", "proofreader_gross_amount_kes", "proofreader_deduction_kes", "proofreader_deduction_reason",
                "proofreader_payout_status", "proofreader_payout_period_id", "proofreaderPaidAt", "quote",
            ):
                out.pop(key, None)
    return out


def _human_bucket():
    if not FIREBASE_ADMIN_SDK_CONFIG_BASE64:
        return None
    try:
        configured = (os.getenv("FIREBASE_STORAGE_BUCKET") or os.getenv("GCS_BUCKET_NAME") or "").strip()
        return firebase_storage.bucket(configured) if configured else firebase_storage.bucket()
    except Exception as exc:
        logger.warning("Human workflow storage is unavailable: %s", exc)
        return None


async def _human_store_upload(job_id: str, upload: UploadFile, folder: str):
    if not upload or not upload.filename:
        return None
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is empty.")
    if len(raw) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"{upload.filename} is larger than 500 MB.")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(upload.filename))[:180] or "attachment"
    path = f"human-workflow/{job_id}/{folder}/{uuid.uuid4().hex}-{safe_name}"
    bucket = _human_bucket()
    if bucket is None:
        raise HTTPException(status_code=503, detail="File storage is not ready yet. Please try again shortly.")
    blob = bucket.blob(path)
    blob.upload_from_string(raw, content_type=upload.content_type or "application/octet-stream")
    return {
        "name": upload.filename,
        "storage_path": path,
        "content_type": upload.content_type or "application/octet-stream",
        "size": len(raw),
    }


async def _human_reclaim_expired_job(job_id: str, job: dict):
    """A worker's TAT deadline passed before they submitted. Take the job
    back from them and return it to the admin queue as "approved" so it can
    be reassigned, exactly like a fresh, unassigned approved job."""
    worker_name = job.get("worker_name") or job.get("worker_email") or "the previous worker"
    now = datetime.now()
    updates = {
        "status": "approved",
        "worker_uid": None,
        "worker_email": None,
        "worker_name": None,
        "assignedAt": None,
        "deadlineAt": None,
        "tat_seconds": None,
        "auto_reassigned_count": int(job.get("auto_reassigned_count") or 0) + 1,
        "last_auto_reassigned_at": now,
        "last_auto_reassigned_worker_name": worker_name,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    try:
        await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, updates)
    except Exception as exc:
        logger.warning("Could not auto-reclaim expired human job %s: %s", job_id, exc)
        return job
    job = dict(job)
    job.update(updates)
    return job


async def _human_check_expiry(job_id: str, job: dict):
    """Return expired work to the admin queue without taking live work away."""
    if not db or not job:
        return job
    now = datetime.now()
    if job.get("split_mode") == "dual":
        segments = [dict(item or {}) for item in (job.get("segments") or [])]
        changed = False
        for item in segments:
            if item.get("status") not in ("assigned", "in_progress"):
                continue
            deadline = _as_dt(item.get("deadlineAt"))
            if not deadline or now <= deadline:
                continue
            previous = item.get("worker_name") or item.get("worker_email") or "the previous worker"
            item.update({
                "status": "available",
                "worker_uid": None,
                "worker_email": None,
                "worker_name": None,
                "assignedAt": None,
                "deadlineAt": None,
                "tat_seconds": None,
                "auto_reassigned_count": int(item.get("auto_reassigned_count") or 0) + 1,
                "last_auto_reassigned_at": now,
                "last_auto_reassigned_worker_name": previous,
            })
            changed = True
        proofreader_status = job.get("proofreader_status")
        proofreader_deadline = _as_dt(job.get("proofreader_deadlineAt"))
        if proofreader_status in ("assigned", "in_progress") and proofreader_deadline and now > proofreader_deadline:
            job = dict(job)
            job.update({"proofreader_status": "available", "proofreader_uid": None, "proofreader_email": None, "proofreader_name": None, "proofreader_deadlineAt": None, "proofreader_tat_seconds": None})
            changed = True
        if changed:
            job = dict(job)
            job["segments"] = segments
            job["assigned_worker_uids"] = list(dict.fromkeys(item.get("worker_uid") for item in segments if item.get("worker_uid")))
            if job.get("proofreader_status") in ("assigned", "in_progress"):
                job["status"] = "proofreading_assigned"
            elif all(item.get("status") == "submitted" for item in segments):
                job["status"] = "proofreading_available"
            else:
                job["status"] = "split_assigned"
            job["updatedAt"] = firestore.SERVER_TIMESTAMP
            try:
                await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {
                    "segments": segments,
                    "assigned_worker_uids": job["assigned_worker_uids"],
                    "proofreader_uid": job.get("proofreader_uid"),
                    "proofreader_email": job.get("proofreader_email"),
                    "proofreader_name": job.get("proofreader_name"),
                    "proofreader_status": job.get("proofreader_status"),
                    "proofreader_deadlineAt": job.get("proofreader_deadlineAt"),
                    "proofreader_tat_seconds": job.get("proofreader_tat_seconds"),
                    "status": job["status"],
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                })
            except Exception as exc:
                logger.warning("Could not update split-job expiry for %s: %s", job_id, exc)
        return job
    if job.get("status") not in ("assigned", "in_progress"):
        return job
    deadline = _as_dt(job.get("deadlineAt"))
    if not deadline or now <= deadline:
        return job
    return await _human_reclaim_expired_job(job_id, job)


async def _human_job(job_id: str):
    if not db:
        raise HTTPException(status_code=503, detail="The workflow database is unavailable.")
    snapshot = await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).get)
    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="That human-transcription job was not found.")
    data = snapshot.to_dict() or {}
    data["id"] = snapshot.id
    data = await _human_check_expiry(job_id, data)
    return data


async def human_expiry_sweep():
    """Background safety net: reclaim any job whose worker TAT deadline has
    passed even if nobody happens to load it in the browser right now."""
    if not db:
        return
    try:
        snapshots = await asyncio.to_thread(
            lambda: list(db.collection(HUMAN_JOB_COLLECTION).where(filter=FieldFilter("status", "in", [
                "assigned", "in_progress", "split_assigned", "split_in_progress",
                "proofreading_assigned", "proofreading_in_progress",
            ])).stream())
        )
    except Exception as exc:
        logger.warning("Human job expiry sweep could not list jobs: %s", exc)
        return
    for snap in snapshots:
        data = snap.to_dict() or {}
        data["id"] = snap.id
        await _human_check_expiry(snap.id, data)


async def human_expiry_monitor():
    """Runs for the life of the app, checking worker TAT deadlines every
    minute so an expired job is returned to the admin queue promptly."""
    while True:
        try:
            await human_expiry_sweep()
        except Exception as exc:
            logger.warning("Human job expiry monitor iteration failed: %s", exc)
        await asyncio.sleep(60)


# ===================== Human worker bi-monthly payouts =====================
# Worker pay accrues across two halves of each calendar month: the 1st-15th
# and the 16th-end of month. When a half ends, every worker's accrued jobs in
# that half are rolled into one pending payout invoice, which an admin later
# marks as paid. The next half starts accruing immediately regardless of
# whether the previous invoice has been paid yet.
HUMAN_PAYOUT_COLLECTION = "human_worker_payouts"
HUMAN_EARNING_ARCHIVE_COLLECTION = "human_worker_earnings"
HUMAN_DEDUCTION_COLLECTION = "human_worker_payment_adjustments"


def _pay_period_bounds(dt):
    """Which half-month period does this datetime fall in, and when does it
    start/end? Returns (period_label, start_dt, end_dt_exclusive)."""
    year, month, day = dt.year, dt.month, dt.day
    last_day = calendar.monthrange(year, month)[1]
    if day <= 15:
        label = f"{year:04d}-{month:02d}-A"
        start = datetime(year, month, 1)
        end = datetime(year, month, 15, 23, 59, 59, 999999)
    else:
        label = f"{year:04d}-{month:02d}-B"
        start = datetime(year, month, 16)
        end = datetime(year, month, last_day, 23, 59, 59, 999999)
    return label, start, end


def _human_worker_earning_items(job_id, job, include_processed=False):
    """Yield normalized earnings for transcription parts and proofreading.

    With include_processed=True this also returns invoiced/paid items, allowing
    payment history and cleanup to use the same stable representation.
    """
    quote = job.get("quote") or {}
    default_rate = int(quote.get("transcriber_payout_kes_per_minute") or HUMAN_STANDARD_PAYOUT_KES)

    def normalize(source, raw, uid, email, name, minutes, completed_at, rate, payout_status, period_id, paid_at):
        if not uid or not completed_at:
            return None
        payout_status = str(payout_status or "").strip().lower()
        if paid_at:
            payout_status = "paid"
        elif period_id and payout_status in ("", "unassigned"):
            payout_status = "invoiced"
        if not include_processed and (payout_status not in ("", "unassigned") or period_id or paid_at):
            return None
        try:
            minutes = max(0, int(minutes or 0))
        except (TypeError, ValueError):
            minutes = 0
        try:
            gross = max(0, int(raw.get("worker_gross_amount_kes") or raw.get("gross_amount_kes") or raw.get("worker_amount_kes") or minutes * rate))
            deduction = max(0, int(raw.get("worker_deduction_kes") or raw.get("deduction_kes") or 0))
        except (TypeError, ValueError):
            gross, deduction = max(0, minutes * rate), 0
        deduction = min(deduction, gross)
        return {
            "source": source, "job_id": job_id, "segment_id": raw.get("id") if source == "segment" else None,
            "worker_uid": uid, "worker_email": email or "", "worker_name": name or "",
            "completed_at": _as_dt(completed_at), "minutes": minutes,
            "gross_amount_kes": gross, "deduction_kes": deduction,
            "deduction_reason": raw.get("worker_deduction_reason") or raw.get("deduction_reason") or "",
            "amount_kes": max(0, gross - deduction), "payout_status": payout_status or "unassigned",
            "payout_period_id": period_id, "paid_at": _as_dt(paid_at) if paid_at else None,
        }

    single = normalize("job", job, job.get("worker_uid"), job.get("worker_email"), job.get("worker_name"), job.get("worker_minutes") or job.get("minutes"), job.get("workerCompletedAt"), default_rate, job.get("payout_status"), job.get("payout_period_id"), job.get("workerPaidAt"))
    if single:
        yield single
    for segment in (job.get("segments") or []):
        item = normalize("segment", segment, segment.get("worker_uid"), segment.get("worker_email"), segment.get("worker_name"), segment.get("worker_minutes") or segment.get("minutes"), segment.get("workerCompletedAt"), default_rate, segment.get("payout_status"), segment.get("payout_period_id"), segment.get("workerPaidAt"))
        if item:
            item["segment_id"] = segment.get("id")
            yield item
    proofreader_financials = dict(job)
    proofreader_financials.update({
        "worker_gross_amount_kes": job.get("proofreader_gross_amount_kes") or job.get("proofreader_amount_kes"),
        "worker_deduction_kes": job.get("proofreader_deduction_kes"),
        "worker_deduction_reason": job.get("proofreader_deduction_reason"),
    })
    proofreader = normalize("proofreader", proofreader_financials, job.get("proofreader_uid"), job.get("proofreader_email"), job.get("proofreader_name"), job.get("proofreader_minutes") or job.get("minutes"), job.get("proofreader_completedAt"), HUMAN_PROOFREADING_PAYOUT_KES, job.get("proofreader_payout_status"), job.get("proofreader_payout_period_id"), job.get("proofreaderPaidAt"))
    if proofreader:
        yield proofreader


async def _close_due_pay_periods():
    """Roll completed job and archived earnings into half-month invoices."""
    if not db:
        return
    now = datetime.now()
    try:
        job_snapshots = await asyncio.to_thread(lambda: list(db.collection(HUMAN_JOB_COLLECTION).stream()))
        archive_snapshots = await asyncio.to_thread(lambda: list(db.collection(HUMAN_EARNING_ARCHIVE_COLLECTION).stream()))
    except Exception as exc:
        logger.warning("Pay period close could not list earnings: %s", exc)
        return
    groups = {}

    def add_item(item, archive_doc_id=None):
        completed_at = _as_dt(item.get("completed_at")) if item.get("completed_at") else None
        if not completed_at:
            return
        label, start_dt, end_dt = _pay_period_bounds(completed_at)
        if now <= end_dt:
            return
        item = dict(item)
        item["completed_at"] = completed_at
        item["archive_doc_id"] = archive_doc_id
        key = (item.get("worker_uid"), label)
        bucket = groups.setdefault(key, {
            "worker_uid": item.get("worker_uid"), "worker_email": item.get("worker_email") or "", "worker_name": item.get("worker_name") or "",
            "period_label": label, "period_start": start_dt, "period_end": end_dt,
            "job_ids": [], "items": [], "total_minutes": 0, "total_amount_kes": 0,
        })
        identity = (item.get("job_id"), item.get("source"), item.get("segment_id"))
        if any((x.get("job_id"), x.get("source"), x.get("segment_id")) == identity for x in bucket["items"]):
            return
        bucket["job_ids"].append(item.get("job_id"))
        bucket["items"].append(item)
        bucket["total_minutes"] += int(item.get("minutes") or 0)
        bucket["total_amount_kes"] += int(item.get("amount_kes") or 0)

    for snap in job_snapshots:
        job = snap.to_dict() or {}
        for item in _human_worker_earning_items(snap.id, job):
            add_item(item)
    for snap in archive_snapshots:
        item = snap.to_dict() or {}
        if str(item.get("payout_status") or "unassigned").lower() not in ("", "unassigned") or item.get("payout_period_id") or item.get("paid_at"):
            continue
        add_item(item, snap.id)

    for (worker_uid, label), bucket in groups.items():
        if not worker_uid:
            continue
        payout_id = f"{worker_uid}_{label}"
        payout_ref = db.collection(HUMAN_PAYOUT_COLLECTION).document(payout_id)
        existing_snapshot = await asyncio.to_thread(payout_ref.get)
        new_items = [{
            "job_id": item["job_id"], "source": item["source"], "segment_id": item.get("segment_id"),
            "minutes": item["minutes"], "amount_kes": item["amount_kes"],
            "gross_amount_kes": item.get("gross_amount_kes", item["amount_kes"]),
            "deduction_kes": item.get("deduction_kes", 0), "deduction_reason": item.get("deduction_reason") or "",
            "completed_at": item.get("completed_at"), "archive_doc_id": item.get("archive_doc_id"),
        } for item in bucket["items"]]
        if existing_snapshot.exists:
            existing = existing_snapshot.to_dict() or {}
            existing_items = existing.get("items") or []
            existing_ids = {(x.get("job_id"), x.get("source"), x.get("segment_id")) for x in existing_items}
            additions = [x for x in new_items if (x.get("job_id"), x.get("source"), x.get("segment_id")) not in existing_ids]
            updates = {
                "job_ids": sorted(set((existing.get("job_ids") or []) + [x for x in bucket["job_ids"] if x])),
                "items": existing_items + additions,
                "total_minutes": int(existing.get("total_minutes") or 0) + sum(int(x["minutes"]) for x in additions),
                "total_amount_kes": int(existing.get("total_amount_kes") or 0) + sum(int(x["amount_kes"]) for x in additions),
                "updatedAt": firestore.SERVER_TIMESTAMP,
            }
            await asyncio.to_thread(payout_ref.set, updates, merge=True)
        else:
            await asyncio.to_thread(payout_ref.set, {
                "worker_uid": worker_uid, "worker_email": bucket["worker_email"], "worker_name": bucket["worker_name"],
                "period_label": bucket["period_label"], "period_start": bucket["period_start"], "period_end": bucket["period_end"],
                "job_ids": sorted(set(x for x in bucket["job_ids"] if x)), "items": new_items,
                "total_minutes": bucket["total_minutes"], "total_amount_kes": bucket["total_amount_kes"], "status": "pending",
                "createdAt": firestore.SERVER_TIMESTAMP, "updatedAt": firestore.SERVER_TIMESTAMP, "paidAt": None, "paidBy": None,
            })
        # Mark only the exact earning as invoiced, wherever its durable copy lives.
        for item in bucket["items"]:
            if item.get("archive_doc_id"):
                await asyncio.to_thread(db.collection(HUMAN_EARNING_ARCHIVE_COLLECTION).document(item["archive_doc_id"]).set, {
                    "payout_status": "invoiced", "payout_period_id": label, "updatedAt": firestore.SERVER_TIMESTAMP,
                }, merge=True)
                continue
            job_ref = db.collection(HUMAN_JOB_COLLECTION).document(item["job_id"])
            snap = await asyncio.to_thread(job_ref.get)
            job = snap.to_dict() or {}
            update = {"updatedAt": firestore.SERVER_TIMESTAMP}
            if item["source"] == "job":
                update.update({"payout_status": "invoiced", "payout_period_id": label})
            elif item["source"] == "proofreader":
                update.update({"proofreader_payout_status": "invoiced", "proofreader_payout_period_id": label})
            elif item["source"] == "segment":
                segments = [dict(x or {}) for x in (job.get("segments") or [])]
                for segment in segments:
                    if segment.get("id") == item.get("segment_id"):
                        segment["payout_status"] = "invoiced"
                        segment["payout_period_id"] = label
                update["segments"] = segments
            if snap.exists:
                await asyncio.to_thread(job_ref.update, update)


async def human_payout_monitor():
    """Runs for the life of the app, closing any half-month pay period that
    has ended so its invoice is ready for the admin without anyone having to
    open the dashboard first."""
    while True:
        try:
            await _close_due_pay_periods()
        except Exception as exc:
            logger.warning("Human payout monitor iteration failed: %s", exc)
        await asyncio.sleep(900)


async def _human_actor(request: Request):
    decoded = _verified_user(request)
    email = (decoded.get("email") or "").strip().lower()
    uid = decoded.get("uid") or ""
    if not uid:
        raise HTTPException(status_code=401, detail="Your account could not be verified.")
    # A human-job admin (real admin, or the dedicated info@typemywordz.ai ops
    # account) gets the "admin" role here even though it is not a full
    # ADMIN_EMAILS admin elsewhere in the app.
    role = "admin" if is_human_job_admin(email) else "client"
    profile = await _load_profile(uid)
    profile = profile or {}
    profile_role = str(profile.get("role") or profile.get("user_type") or "").strip().lower()
    # A trainee belongs in the private Training Room until an admin explicitly
    # promotes them.  Only approved workers may see assigned human jobs.
    if profile.get("workerApproved") or profile_role in {"worker", "transcriber"}:
        role = "worker"
    return {"uid": uid, "email": email, "role": role, "profile": profile}


async def _human_assert_access(job, actor, allow_admin=True):
    if actor["role"] == "admin" and allow_admin:
        return
    if job.get("client_uid") == actor["uid"] or job.get("worker_uid") == actor["uid"]:
        return
    if any((item or {}).get("worker_uid") == actor["uid"] for item in (job.get("segments") or [])):
        return
    if job.get("proofreader_uid") == actor["uid"]:
        return
    raise HTTPException(status_code=403, detail="You do not have access to this job.")


@app.post("/human-transcription/jobs")
async def human_create_job(
    request: Request,
    audio: UploadFile = File(None),
    attachments: List[UploadFile] = File(default=[]),
    seconds: float = Form(0),
    turnaround: str = Form("standard"),
    difficulty: str = Form("standard"),
    timestamps: bool = Form(True),
    speakers: str = Form("1-2"),
    speaker_labels: bool = Form(True),
    instructions: str = Form(""),
    service: str = Form("standard"),
    formatting: str = Form("standard"),
    source_type: str = Form("human_transcription"),
    initial_transcript: str = Form(""),
    client_request_id: str = Form(""),
):
    actor = await _human_actor(request)
    if actor["role"] != "client" and actor["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only a client can request human work.")
    if seconds <= 0:
        raise HTTPException(status_code=400, detail="The recording length is required.")
    source_type = str(source_type or "human_transcription").strip().lower()
    if source_type not in {"human_transcription", "ai_proofreading"}:
        raise HTTPException(status_code=400, detail="That proofreading source is not supported.")
    if source_type == "human_transcription" and (not audio or not audio.filename):
        raise HTTPException(status_code=400, detail="An audio or video file is required for a new human transcript.")
    if source_type == "ai_proofreading" and not str(initial_transcript or "").strip():
        raise HTTPException(status_code=400, detail="The AI transcript is required for proofreading.")
    client_request_id = str(client_request_id or "").strip()[:120]
    # A timed-out browser request can still finish creating the job. Reusing
    # the same client request id makes a retry return that job instead of
    # creating a duplicate order.
    if client_request_id and db is not None:
        existing_snapshots = await asyncio.to_thread(
            lambda: list(db.collection(HUMAN_JOB_COLLECTION)
                         .where(filter=FieldFilter("client_uid", "==", actor["uid"]))
                         .limit(100).stream())
        )
        for existing_snapshot in existing_snapshots:
            existing_job = existing_snapshot.to_dict() or {}
            if str(existing_job.get("client_request_id") or "") == client_request_id:
                existing_job["id"] = existing_snapshot.id
                return {"job": _human_public(existing_job), "reservation": "not_created", "credits_deducted": 0, "already_created": True}
    quote = human_credit_quote(seconds, turnaround, difficulty, service, speakers, timestamps, formatting)
    profile = await _load_profile(actor["uid"])
    balance = read_balance(profile or {})
    # A real admin, or the dedicated human-job-admin account, never needs
    # credits for a human-transcription job it owns. Everyone else -- every
    # ordinary client, including info@typemywordz.ai's own AI-transcription
    # and Ask TypeMyworDz usage elsewhere in the app -- still pays normally.
    exempt = credits_exempt(actor["email"]) or human_job_credits_exempt(actor["email"])
    if not exempt and balance["spendable"] < quote["credits"]:
        raise HTTPException(status_code=409, detail=f"You need {quote['credits'] - balance['spendable']} more credits before human work can begin.")
    job_id = uuid.uuid4().hex
    audio_meta = await _human_store_upload(job_id, audio, "audio") if audio and audio.filename else None
    attachment_meta = []
    for item in attachments or []:
        attachment_meta.append(await _human_store_upload(job_id, item, "instructions"))
    now = firestore.SERVER_TIMESTAMP
    job = {
        "client_uid": actor["uid"],
        "client_email": actor["email"],
        "client_request_id": client_request_id,
        "status": "pending_admin",
        "createdAt": now,
        "updatedAt": now,
        "seconds": float(seconds),
        "minutes": quote["minutes"],
        "turnaround": turnaround,
        "difficulty": difficulty,
        "service": service,
        "formatting": formatting,
        "timestamps": bool(timestamps),
        "speakers": speakers,
        "speaker_labels": bool(speaker_labels),
        "source_type": source_type,
        "instructions": (instructions or "").strip()[:12000],
        "audio": audio_meta,
        "instruction_attachments": attachment_meta,
        "quote_credits": int(quote["credits"]),
        "quote": quote,
        "worker_uid": None,
        "worker_email": None,
        "worker_name": None,
        "transcript": str(initial_transcript or "")[:1000000] if source_type == "ai_proofreading" else "",
        "worker_notes": "",
        "final_attachment": None,
        "admin_feedback": "",
        "worker_rating": None,
        "credits_charged": 0,
        "releasedAt": None,
        # TAT/deadline tracking (set when the job is assigned to a worker).
        "assignedAt": None,
        "deadlineAt": None,
        "tat_seconds": None,
        "auto_reassigned_count": 0,
        "last_auto_reassigned_at": None,
        "last_auto_reassigned_worker_name": None,
        # Worker pay/payout tracking.
        "workerCompletedAt": None,
        "worker_minutes": None,
        "worker_amount_kes": None,
        "payout_status": None,
        "payout_period_id": None,
        "workerPaymentStatus": None,
        "workerPaidAt": None,
        # Optional two-worker workflow. The parent job remains the client-facing
        # record; each segment carries its own worker, deadline and submission.
        "split_mode": "single",
        "segments": [],
        "assigned_worker_uids": [],
        "proofreader_uid": None,
        "proofreader_email": None,
        "proofreader_name": None,
        "proofreader_status": None,
        "proofreader_deadlineAt": None,
        "proofreader_tat_seconds": None,
        "proofreader_completedAt": None,
        "proofreader_minutes": None,
        "proofreader_amount_kes": None,
        "proofreader_payout_status": None,
        "proofreader_payout_period_id": None,
    }
    if db is None:
        raise HTTPException(status_code=503, detail="The workflow database is unavailable.")
    job_ref = db.collection(HUMAN_JOB_COLLECTION).document(job_id)
    await asyncio.to_thread(job_ref.set, job)
    # Resolve Firestore server timestamps before returning the created job.
    # Otherwise a successful upload can look like a failed request in the browser.
    saved_snapshot = await asyncio.to_thread(job_ref.get)
    saved_job = saved_snapshot.to_dict() or job
    saved_job["id"] = job_id
    return {"job": _human_public(saved_job), "reservation": "not_created", "credits_deducted": 0}


@app.get("/human-transcription/jobs")
async def human_list_jobs(request: Request, scope: str = "mine"):
    actor = await _human_actor(request)
    if not db:
        return {"jobs": []}
    ref = db.collection(HUMAN_JOB_COLLECTION)
    if actor["role"] == "admin" or scope == "admin":
        snapshots = await asyncio.to_thread(lambda: list(ref.order_by("createdAt", direction=firestore.Query.DESCENDING).limit(100).stream()))
    elif actor["role"] == "worker" or scope in {"assigned", "finished"}:
        # Keep the original single-worker query and add the split parent query.
        found = {}
        for snap in await asyncio.to_thread(lambda: list(ref.where(filter=FieldFilter("worker_uid", "==", actor["uid"])).stream())):
            found[snap.id] = snap
        for snap in await asyncio.to_thread(lambda: list(ref.where(filter=FieldFilter("assigned_worker_uids", "array_contains", actor["uid"])).stream())):
            found[snap.id] = snap
        for snap in await asyncio.to_thread(lambda: list(ref.where(filter=FieldFilter("proofreader_uid", "==", actor["uid"])).stream())):
            found[snap.id] = snap
        snapshots = list(found.values())
    else:
        snapshots = await asyncio.to_thread(lambda: list(ref.where(filter=FieldFilter("client_uid", "==", actor["uid"])).stream()))
    jobs = []
    view_role = "human_ops_admin" if actor["role"] == "admin" and not is_admin_user(actor.get("email") or "") else actor["role"]
    for snap in snapshots:
        item = snap.to_dict() or {}
        item["id"] = snap.id
        item = await _human_check_expiry(snap.id, item)
        if actor["role"] == "worker" or scope in {"assigned", "finished"}:
            assignment = _human_worker_segment(item.get("segments") or [], actor["uid"])
            is_proofreader = item.get("proofreader_uid") == actor["uid"]
            current_status = item.get("proofreader_status") if is_proofreader else (assignment.get("status") if assignment else item.get("status"))
            finished = {"submitted", "client_review", "client_approved", "released"}
            active = {"assigned", "in_progress"}
            if scope == "finished" and current_status not in finished:
                continue
            if scope != "finished" and current_status not in active:
                continue
        jobs.append(_human_public_for(item, view_role, actor.get("uid") or ""))
    jobs.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    response = {"jobs": jobs}
    if actor["role"] == "worker":
        ratings = []
        for snapshot in snapshots:
            raw_rating = (snapshot.to_dict() or {}).get("worker_rating")
            try:
                value = float(raw_rating)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and 1 <= value <= 5:
                ratings.append(value)
        response["worker_rating_summary"] = {
            "average": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "count": len(ratings),
        }
    return response


@app.get("/human-transcription/notifications")
async def human_workflow_notifications(request: Request, since: str = ""):
    """Return small, role-filtered human-work events for app-wide alerts."""
    actor = await _human_actor(request)
    if not db:
        return {"events": [], "server_time": datetime.now().astimezone().isoformat()}
    try:
        since_dt = _as_dt(datetime.fromisoformat(str(since).replace("Z", "+00:00"))) if since else None
    except Exception:
        since_dt = None
    if not since_dt:
        since_dt = datetime.now() - timedelta(seconds=10)
    ref = db.collection(HUMAN_JOB_COLLECTION)
    found = {}
    if actor["role"] == "admin":
        snapshots = await asyncio.to_thread(lambda: list(ref.where(filter=FieldFilter("updatedAt", ">=", since_dt)).stream()))
        found.update({snap.id: snap for snap in snapshots})
    elif actor["role"] == "worker":
        for field, op in (("worker_uid", "=="), ("assigned_worker_uids", "array_contains"), ("proofreader_uid", "==")):
            snapshots = await asyncio.to_thread(lambda field=field, op=op: list(ref.where(filter=FieldFilter(field, op, actor["uid"])).stream()))
            found.update({snap.id: snap for snap in snapshots})
        takebacks = await asyncio.to_thread(lambda: list(ref.where(filter=FieldFilter("last_assignment_takeback_at", ">=", since_dt)).stream()))
        found.update({snap.id: snap for snap in takebacks})
    else:
        snapshots = await asyncio.to_thread(lambda: list(ref.where(filter=FieldFilter("client_uid", "==", actor["uid"])).stream()))
        found.update({snap.id: snap for snap in snapshots})

    def recent(value):
        moment = _as_dt(value) if value else None
        return bool(moment and moment >= since_dt)

    events = []
    for snap in found.values():
        job = snap.to_dict() or {}
        job_id = snap.id
        audio = job.get("audio") or {}
        job_name = str(audio.get("name") or job.get("audio_name") or f"Human job {job_id[:8]}")[:180]
        status = str(job.get("status") or "")
        if actor["role"] == "admin":
            if status == "pending_admin" and recent(job.get("createdAt")):
                events.append({"type": "new_request", "job_id": job_id, "job_name": job_name, "status": status, "updated_at": _human_iso(job.get("createdAt"))})
            if recent(job.get("last_auto_reassigned_at")):
                events.append({"type": "returned_to_queue", "job_id": job_id, "job_name": job_name, "updated_at": _human_iso(job.get("last_auto_reassigned_at"))})
            for part in (job.get("segments") or []):
                if recent(part.get("last_auto_reassigned_at")):
                    events.append({"type": "returned_to_queue", "job_id": job_id, "job_name": job_name, "label": part.get("label") or "A job part", "updated_at": _human_iso(part.get("last_auto_reassigned_at")), "event_id": part.get("id")})
                if part.get("status") == "submitted" and recent(part.get("workerCompletedAt")):
                    events.append({"type": "submission", "job_id": job_id, "job_name": job_name, "label": part.get("label") or "A job part", "updated_at": _human_iso(part.get("workerCompletedAt")), "event_id": part.get("id")})
            if job.get("proofreader_status") == "submitted" and recent(job.get("proofreader_completedAt")):
                events.append({"type": "submission", "job_id": job_id, "job_name": job_name, "label": "Final proofreading", "updated_at": _human_iso(job.get("proofreader_completedAt")), "event_id": "proofreader"})
            if not job.get("split_mode") and status in {"submitted", "client_review"} and recent(job.get("workerCompletedAt")):
                events.append({"type": "submission", "job_id": job_id, "job_name": job_name, "label": "Transcript", "updated_at": _human_iso(job.get("workerCompletedAt")), "event_id": "job"})
        elif actor["role"] == "client":
            if status == "client_review" and recent(job.get("reviewedAt")):
                events.append({"type": "review", "job_id": job_id, "job_name": job_name, "status": status, "updated_at": _human_iso(job.get("reviewedAt"))})
            if status == "released" and recent(job.get("releasedAt")):
                events.append({"type": "released", "job_id": job_id, "job_name": job_name, "status": status, "updated_at": _human_iso(job.get("releasedAt"))})
        else:
            assignments = []
            if job.get("worker_uid") == actor["uid"] and job.get("status") in {"assigned", "in_progress"}:
                assignments.append(("transcriber", job.get("assignedAt")))
            for part in (job.get("segments") or []):
                if part.get("worker_uid") == actor["uid"] and part.get("status") in {"assigned", "in_progress"}:
                    assignments.append((part.get("label") or "transcriber", part.get("assignedAt")))
            if job.get("proofreader_uid") == actor["uid"] and job.get("proofreader_status") in {"assigned", "in_progress"}:
                assignments.append(("proofreader", job.get("proofreader_assignedAt")))
            for role, assigned_at in assignments:
                if recent(assigned_at):
                    events.append({"type": "assignment", "job_id": job_id, "job_name": job_name, "label": role, "updated_at": _human_iso(assigned_at)})
            takeback_at = job.get("last_assignment_takeback_at")
            if job.get("last_assignment_takeback_worker_uid") == actor["uid"] and recent(takeback_at):
                events.append({
                    "type": "assignment_taken_back", "job_id": job_id, "job_name": job_name,
                    "label": job.get("last_assignment_takeback_label") or "Your assignment",
                    "updated_at": _human_iso(takeback_at), "event_id": _human_iso(takeback_at),
                })
        message_summaries = job.get("last_message_by_thread") or {}
        if not message_summaries and job.get("last_message"):
            legacy = job.get("last_message") or {}
            message_summaries = {legacy.get("thread") or "client": legacy}
        visible_threads = ("client", "worker") if actor["role"] == "admin" else (("client",) if actor["role"] == "client" else (("worker",) if actor["role"] == "worker" else ()))
        for message_thread in visible_threads:
            last_message = message_summaries.get(message_thread) or {}
            if last_message.get("sender_uid") != actor["uid"] and recent(last_message.get("createdAt")):
                events.append({"type": "message", "job_id": job_id, "job_name": job_name, "sender_role": last_message.get("sender_role"), "thread": message_thread, "message_id": last_message.get("id"), "updated_at": _human_iso(last_message.get("createdAt"))})
    events.sort(key=lambda event: str(event.get("updated_at") or ""))
    return {"events": events, "server_time": datetime.now().astimezone().isoformat()}


@app.get("/human-transcription/worker/payment-history")
async def human_worker_payment_history(request: Request):
    actor = await _human_actor(request)
    if actor["role"] != "worker":
        raise HTTPException(status_code=403, detail="Worker access is required.")
    if not db:
        return {"paid": [], "upcoming": [], "totals": {"paid_kes": 0, "upcoming_kes": 0}, "current_period": None, "pending_payouts": []}
    await _close_due_pay_periods()
    found = {}
    for snap in await asyncio.to_thread(lambda: list(db.collection(HUMAN_JOB_COLLECTION).where(filter=FieldFilter("worker_uid", "==", actor["uid"])).stream())):
        found[snap.id] = snap
    for snap in await asyncio.to_thread(lambda: list(db.collection(HUMAN_JOB_COLLECTION).where(filter=FieldFilter("assigned_worker_uids", "array_contains", actor["uid"])).stream())):
        found[snap.id] = snap
    for snap in await asyncio.to_thread(lambda: list(db.collection(HUMAN_JOB_COLLECTION).where(filter=FieldFilter("proofreader_uid", "==", actor["uid"])).stream())):
        found[snap.id] = snap
    paid, upcoming = [], []
    seen_earnings = set()

    def append_payment_row(item, job_status=None, archive=False):
        if item.get("worker_uid") != actor["uid"]:
            return
        identity = (item.get("job_id"), item.get("source"), item.get("segment_id"))
        if identity in seen_earnings:
            return
        seen_earnings.add(identity)
        role = "Proofreader" if item.get("source") == "proofreader" else ("Part " + str(item.get("segment_id")).split("_")[-1] if item.get("source") == "segment" and item.get("segment_id") else "Transcriber")
        payout_status = item.get("payout_status") or "accruing"
        if payout_status == "unassigned":
            payout_status = "accruing"
        is_paid = payout_status == "paid" or bool(item.get("paid_at"))
        minutes = int(item.get("minutes") or 0)
        amount = int(item.get("amount_kes") or 0)
        row = {
            "job_id": item.get("job_id"), "role": role, "status": "paid" if is_paid else "upcoming",
            "job_status": job_status or item.get("job_status"), "payout_status": "paid" if is_paid else payout_status,
            "payout_period_id": item.get("payout_period_id"), "minutes": minutes, "amount_kes": amount,
            "gross_amount_kes": int(item.get("gross_amount_kes") or amount), "deduction_kes": int(item.get("deduction_kes") or 0),
            "deduction_reason": item.get("deduction_reason") or "", "rate_kes_per_minute": round(amount / minutes, 2) if minutes else 0,
            "completed_at": _human_iso(item.get("completed_at")), "paid_at": _human_iso(item.get("paid_at")), "archived": archive,
        }
        (paid if is_paid else upcoming).append(row)

    for snap in found.values():
        job = snap.to_dict() or {}
        for item in _human_worker_earning_items(snap.id, job, include_processed=True):
            append_payment_row(item, job.get("status"))
    archive_snapshots = await asyncio.to_thread(lambda: list(db.collection(HUMAN_EARNING_ARCHIVE_COLLECTION).where(filter=FieldFilter("worker_uid", "==", actor["uid"])).stream()))
    for snap in archive_snapshots:
        item = snap.to_dict() or {}
        append_payment_row(item, item.get("job_status"), archive=True)
    paid.sort(key=lambda item: str(item.get("paid_at") or item.get("completed_at") or ""), reverse=True)
    upcoming.sort(key=lambda item: str(item.get("completed_at") or ""), reverse=True)
    label, start_dt, end_dt = _pay_period_bounds(datetime.now())
    current_accrued = [item for item in upcoming if item.get("payout_status") in (None, "accruing", "unassigned") and item.get("payout_period_id") in (None, label)]
    is_proofreading = lambda row: row.get("role") == "Proofreader"
    def earning_breakdown(rows):
        transcription_rows = [row for row in rows if not is_proofreading(row)]
        proofreading_rows = [row for row in rows if is_proofreading(row)]
        return {
            "transcription_kes": sum(row["amount_kes"] for row in transcription_rows),
            "transcription_minutes": sum(row["minutes"] for row in transcription_rows),
            "proofreading_kes": sum(row["amount_kes"] for row in proofreading_rows),
            "proofreading_minutes": sum(row["minutes"] for row in proofreading_rows),
            "total_kes": sum(row["amount_kes"] for row in rows),
            "total_minutes": sum(row["minutes"] for row in rows),
        }
    paid_breakdown = earning_breakdown(paid)
    upcoming_breakdown = earning_breakdown(upcoming)
    current_breakdown = earning_breakdown(current_accrued)
    payout_snapshots = await asyncio.to_thread(lambda: list(db.collection(HUMAN_PAYOUT_COLLECTION).where(filter=FieldFilter("worker_uid", "==", actor["uid"])).stream()))
    pending_payouts = []
    for snap in payout_snapshots:
        payout = snap.to_dict() or {}
        if payout.get("status") == "paid":
            continue
        payout_items = payout.get("items") or []
        transcription_items = [item for item in payout_items if item.get("source") != "proofreader"]
        proofreading_items = [item for item in payout_items if item.get("source") == "proofreader"]
        pending_payouts.append({
            "payout_id": snap.id,
            "period_label": payout.get("period_label"),
            "period_start": _human_iso(payout.get("period_start")),
            "period_end": _human_iso(payout.get("period_end")),
            "total_minutes": int(payout.get("total_minutes") or 0),
            "total_amount_kes": int(payout.get("total_amount_kes") or 0),
            "total_deduction_kes": sum(int(item.get("deduction_kes") or 0) for item in payout_items),
            "gross_total_kes": int(payout.get("total_amount_kes") or 0) + sum(int(item.get("deduction_kes") or 0) for item in payout_items),
            "transcription_amount_kes": sum(int(item.get("amount_kes") or 0) for item in transcription_items),
            "proofreading_amount_kes": sum(int(item.get("amount_kes") or 0) for item in proofreading_items),
            "status": payout.get("status") or "pending",
        })
        if not payout_items:
            pending_payouts[-1]["transcription_amount_kes"] = int(payout.get("total_amount_kes") or 0)
    pending_payouts.sort(key=lambda item: str(item.get("period_label") or ""), reverse=True)
    return {
        "paid": paid,
        "upcoming": upcoming,
        "totals": {
            "paid_kes": paid_breakdown["total_kes"],
            "upcoming_kes": upcoming_breakdown["total_kes"],
            "paid_transcription_kes": paid_breakdown["transcription_kes"],
            "paid_proofreading_kes": paid_breakdown["proofreading_kes"],
            "upcoming_transcription_kes": upcoming_breakdown["transcription_kes"],
            "upcoming_proofreading_kes": upcoming_breakdown["proofreading_kes"],
        },
        "current_period": {
            "label": label, "start": start_dt.isoformat(), "end": end_dt.isoformat(),
            "accrued_kes": current_breakdown["total_kes"],
            "accrued_minutes": current_breakdown["total_minutes"],
            "transcription_kes": current_breakdown["transcription_kes"],
            "transcription_minutes": current_breakdown["transcription_minutes"],
            "proofreading_kes": current_breakdown["proofreading_kes"],
            "proofreading_minutes": current_breakdown["proofreading_minutes"],
        },
        "pending_payouts": pending_payouts,
    }


@app.get("/human-transcription/workers")
async def human_list_workers(request: Request):
    _require_human_job_admin(request)
    if not db:
        return {"workers": []}
    workers = []
    for snap in await asyncio.to_thread(lambda: list(db.collection("users").stream())):
        data = snap.to_dict() or {}
        role = str(data.get("role") or data.get("user_type") or "").strip().lower()
        # Paid trainees stay in Training Room; only explicitly approved
        # workers should appear in the assignment list.
        if not data.get("workerApproved") and role not in {"worker", "transcriber"}:
            continue
        workers.append({
            "uid": data.get("uid") or snap.id,
            "email": data.get("email") or "",
            "name": data.get("name") or data.get("full_name") or data.get("displayName") or "Unnamed worker",
            "role": role or "worker",
            "approved": bool(data.get("workerApproved") or role == "worker"),
            "available": data.get("is_available", True),
        })
    return {"workers": workers}


@app.get("/human-transcription/worker/payment-profile")
async def human_worker_payment_profile(request: Request):
    actor = await _human_actor(request)
    if actor["role"] != "worker":
        raise HTTPException(status_code=403, detail="Worker access is required.")
    profile = actor.get("profile") or {}
    return {
        "official_id_name": profile.get("officialIdName") or profile.get("name") or "",
        "mpesa_registered_name": profile.get("mpesaRegisteredName") or "",
        "mpesa_number": profile.get("mpesaNumber") or "",
        "complete": bool(profile.get("mpesaRegisteredName") and profile.get("mpesaNumber")),
    }


@app.post("/human-transcription/worker/payment-profile")
async def human_worker_save_payment_profile(request: Request):
    actor = await _human_actor(request)
    if actor["role"] != "worker":
        raise HTTPException(status_code=403, detail="Worker access is required.")
    payload = await request.json()
    registered_name = re.sub(r"\s+", " ", str(payload.get("mpesa_registered_name") or "").strip())[:120]
    raw_number = re.sub(r"[\s()+.-]", "", str(payload.get("mpesa_number") or ""))
    if bool(registered_name) != bool(raw_number):
        raise HTTPException(status_code=400, detail="Enter both the M-Pesa registered name and number, or clear both fields.")
    normalized_number = ""
    if registered_name:
        if len(registered_name) < 2:
            raise HTTPException(status_code=400, detail="Enter the M-Pesa account name as it appears in M-Pesa.")
        if not re.fullmatch(r"(?:254|0)(?:7|1)\d{8}", raw_number):
            raise HTTPException(status_code=400, detail="Enter a valid Kenyan M-Pesa number, such as 0712 345 678 or 254712345678.")
        normalized_number = "254" + raw_number[1:] if raw_number.startswith("0") else raw_number
    updates = {
        "mpesaRegisteredName": registered_name,
        "mpesaNumber": normalized_number,
        "mpesaDetailsUpdatedAt": firestore.SERVER_TIMESTAMP,
    }
    await asyncio.to_thread(db.collection("users").document(actor["uid"]).set, updates, merge=True)
    return {"saved": True, "official_id_name": (actor.get("profile") or {}).get("officialIdName") or (actor.get("profile") or {}).get("name") or "", "mpesa_registered_name": registered_name, "mpesa_number": normalized_number, "complete": bool(registered_name and normalized_number)}


@app.get("/human-transcription/admin/workers/{worker_uid}/payment-profile")
async def human_admin_worker_payment_profile(worker_uid: str, request: Request):
    admin = _require_admin(request)
    if not db:
        raise HTTPException(status_code=503, detail="The workflow database is unavailable.")
    profile = await _load_profile(worker_uid)
    if profile is None:
        raise HTTPException(status_code=404, detail="The worker account was not found.")
    role = str(profile.get("role") or profile.get("user_type") or "").strip().lower()
    if not profile.get("workerApproved") and role not in {"worker", "transcriber"}:
        raise HTTPException(status_code=404, detail="The approved worker account was not found.")
    logger.info("Human-work admin %s viewed payout details for worker %s", (admin.get("email") or "").lower(), worker_uid)
    return {
        "worker_uid": worker_uid,
        "worker_name": profile.get("name") or profile.get("displayName") or profile.get("full_name") or "",
        "worker_email": profile.get("email") or "",
        "official_id_name": profile.get("officialIdName") or profile.get("name") or "",
        "mpesa_registered_name": profile.get("mpesaRegisteredName") or "",
        "mpesa_number": profile.get("mpesaNumber") or "",
        "complete": bool(profile.get("mpesaRegisteredName") and profile.get("mpesaNumber")),
    }


@app.get("/api/admin/worker-payments/search")
async def admin_worker_payments_search(request: Request, worker_uid: str = "", start_date: str = "", end_date: str = "", status: str = "all"):
    _require_admin(request)
    if not db:
        return {"jobs": [], "total_minutes": 0, "total_amount_kes": 0}
    await _close_due_pay_periods()
    def _parse_day(value, end_of_day=False):
        if not value:
            return None
        try:
            d = datetime.fromisoformat(value[:10])
        except Exception:
            return None
        return d.replace(hour=23, minute=59, second=59, microsecond=999999) if end_of_day else d
    start_bound = _parse_day(start_date)
    end_bound = _parse_day(end_date, end_of_day=True)
    status = (status or "all").strip().lower()
    rows, seen = [], set()

    def append_admin_payment(item, job_status=None):
        if worker_uid and item.get("worker_uid") != worker_uid:
            return
        identity = (item.get("job_id"), item.get("source"), item.get("segment_id"))
        if identity in seen:
            return
        completed_at = _as_dt(item.get("completed_at")) if item.get("completed_at") else None
        if start_bound and (not completed_at or completed_at < start_bound):
            return
        if end_bound and (not completed_at or completed_at > end_bound):
            return
        payout_status = item.get("payout_status") or "accruing"
        if payout_status == "unassigned":
            payout_status = "accruing"
        if item.get("paid_at"):
            payout_status = "paid"
        if status != "all" and payout_status != status:
            return
        role = "Proofreader" if item.get("source") == "proofreader" else ("Part " + str(item.get("segment_id")).split("_")[-1] if item.get("source") == "segment" and item.get("segment_id") else "Transcriber")
        amount = int(item.get("amount_kes") or 0)
        seen.add(identity)
        rows.append({
            "job_id": item.get("job_id"), "source": item.get("source"), "segment_id": item.get("segment_id"),
            "role": role, "worker_uid": item.get("worker_uid"),
            "worker_email": item.get("worker_email") or "", "worker_name": item.get("worker_name") or "",
            "minutes": int(item.get("minutes") or 0), "amount_kes": amount,
            "gross_amount_kes": int(item.get("gross_amount_kes") or amount), "deduction_kes": int(item.get("deduction_kes") or 0),
            "deduction_reason": item.get("deduction_reason") or "", "payout_status": payout_status,
            "payout_period_id": item.get("payout_period_id"), "completed_at": _human_iso(completed_at),
            "paid_at": _human_iso(item.get("paid_at")), "job_status": job_status or item.get("job_status"),
        })

    for snap in await asyncio.to_thread(lambda: list(db.collection(HUMAN_JOB_COLLECTION).stream())):
        job = snap.to_dict() or {}
        for item in _human_worker_earning_items(snap.id, job, include_processed=True):
            append_admin_payment(item, job.get("status"))
    archive_query = db.collection(HUMAN_EARNING_ARCHIVE_COLLECTION)
    archive_snapshots = await asyncio.to_thread(lambda: list(archive_query.where(filter=FieldFilter("worker_uid", "==", worker_uid)).stream())) if worker_uid else await asyncio.to_thread(lambda: list(archive_query.stream()))
    for snap in archive_snapshots:
        item = snap.to_dict() or {}
        append_admin_payment(item, item.get("job_status"))
    rows.sort(key=lambda item: str(item.get("completed_at") or ""), reverse=True)
    transcription_rows = [item for item in rows if item["role"] != "Proofreader"]
    proofreading_rows = [item for item in rows if item["role"] == "Proofreader"]
    return {
        "jobs": rows,
        "total_minutes": sum(item["minutes"] for item in rows),
        "total_amount_kes": sum(item["amount_kes"] for item in rows),
        "total_deduction_kes": sum(item["deduction_kes"] for item in rows),
        "gross_total_kes": sum(item["gross_amount_kes"] for item in rows),
        "transcription_amount_kes": sum(item["amount_kes"] for item in transcription_rows),
        "proofreading_amount_kes": sum(item["amount_kes"] for item in proofreading_rows),
        "transcription_minutes": sum(item["minutes"] for item in transcription_rows),
        "proofreading_minutes": sum(item["minutes"] for item in proofreading_rows),
        "job_count": len(rows),
    }


@app.get("/api/admin/worker-payouts")
async def admin_worker_payouts(request: Request, worker_uid: str = "", status: str = "all"):
    """List worker payment invoices; only the main admin may view them."""
    _require_admin(request)
    if not db:
        return {"payouts": [], "totals": {"pending_kes": 0, "paid_kes": 0}}
    await _close_due_pay_periods()
    ref = db.collection(HUMAN_PAYOUT_COLLECTION)
    if worker_uid:
        snapshots = await asyncio.to_thread(lambda: list(ref.where(filter=FieldFilter("worker_uid", "==", worker_uid)).stream()))
    else:
        snapshots = await asyncio.to_thread(lambda: list(ref.stream()))
    status = (status or "all").strip().lower()
    payouts = []
    pending_total, paid_total = 0, 0
    for snap in snapshots:
        payout = snap.to_dict() or {}
        payout_status = payout.get("status") or "pending"
        amount = int(payout.get("total_amount_kes") or 0)
        if payout_status == "paid":
            paid_total += amount
        else:
            pending_total += amount
        if status != "all" and payout_status != status:
            continue
        invoice_items = payout.get("items") or []
        transcription_items = [item for item in invoice_items if item.get("source") != "proofreader"]
        proofreading_items = [item for item in invoice_items if item.get("source") == "proofreader"]
        payouts.append({
            "payout_id": snap.id,
            "worker_uid": payout.get("worker_uid"),
            "worker_email": payout.get("worker_email") or "",
            "worker_name": payout.get("worker_name") or "",
            "period_label": payout.get("period_label"),
            "period_start": _human_iso(payout.get("period_start")),
            "period_end": _human_iso(payout.get("period_end")),
            "total_minutes": int(payout.get("total_minutes") or 0),
            "total_amount_kes": amount,
            "total_deduction_kes": sum(int(item.get("deduction_kes") or 0) for item in invoice_items),
            "gross_total_kes": amount + sum(int(item.get("deduction_kes") or 0) for item in invoice_items),
            "transcription_minutes": sum(int(item.get("minutes") or 0) for item in transcription_items),
            "transcription_amount_kes": sum(int(item.get("amount_kes") or 0) for item in transcription_items),
            "proofreading_minutes": sum(int(item.get("minutes") or 0) for item in proofreading_items),
            "proofreading_amount_kes": sum(int(item.get("amount_kes") or 0) for item in proofreading_items),
            "status": payout_status,
            "job_ids": payout.get("job_ids") or [],
            "paid_at": _human_iso(payout.get("paidAt")),
            "paid_by": payout.get("paidBy"),
        })
        if not invoice_items:
            payouts[-1]["transcription_minutes"] = int(payout.get("total_minutes") or 0)
            payouts[-1]["transcription_amount_kes"] = amount
    payouts.sort(key=lambda item: str(item.get("period_label") or ""), reverse=True)
    return {"payouts": payouts, "totals": {"pending_kes": pending_total, "paid_kes": paid_total}}


@app.post("/api/admin/worker-payouts/{payout_id}/mark-paid")
async def admin_mark_payout_paid(payout_id: str, request: Request):
    """Main admin confirms a half-month invoice has actually been paid out."""
    admin = _require_admin(request)
    if not db:
        raise HTTPException(status_code=503, detail="The workflow database is unavailable.")
    payout_ref = db.collection(HUMAN_PAYOUT_COLLECTION).document(payout_id)
    snapshot = await asyncio.to_thread(payout_ref.get)
    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="That payout invoice was not found.")
    payout = snapshot.to_dict() or {}
    if payout.get("status") == "paid":
        raise HTTPException(status_code=409, detail="This payout has already been marked as paid.")
    now = datetime.now()
    await asyncio.to_thread(payout_ref.set, {"status": "paid", "paidAt": now, "paidBy": (admin.get("email") or "").lower(), "updatedAt": firestore.SERVER_TIMESTAMP}, merge=True)
    items = payout.get("items") or []
    job_ids = payout.get("job_ids") or []
    if items:
        for item in items:
            job_id = item.get("job_id")
            source = item.get("source") or "job"
            segment_id = item.get("segment_id")
            archive_id = item.get("archive_doc_id") or _human_archive_id(job_id, source, segment_id)
            archive_ref = db.collection(HUMAN_EARNING_ARCHIVE_COLLECTION).document(archive_id)
            archive_snapshot = await asyncio.to_thread(archive_ref.get)
            if item.get("archive_doc_id") or archive_snapshot.exists:
                await asyncio.to_thread(archive_ref.set, {"payout_status": "paid", "paid_at": now, "updatedAt": firestore.SERVER_TIMESTAMP}, merge=True)
                continue
            job_ref = db.collection(HUMAN_JOB_COLLECTION).document(job_id)
            snap = await asyncio.to_thread(job_ref.get)
            if not snap.exists:
                continue
            job = snap.to_dict() or {}
            update = {"updatedAt": firestore.SERVER_TIMESTAMP}
            if source == "job":
                update.update({"payout_status": "paid", "workerPaymentStatus": "paid", "workerPaidAt": now})
            elif source == "proofreader":
                update.update({"proofreader_payout_status": "paid", "proofreaderPaidAt": now})
            elif source == "segment":
                segments = [dict(x or {}) for x in (job.get("segments") or [])]
                for segment in segments:
                    if segment.get("id") == segment_id:
                        segment.update({"payout_status": "paid", "workerPaymentStatus": "paid", "workerPaidAt": now})
                update["segments"] = segments
            await asyncio.to_thread(job_ref.update, update)
    elif job_ids:
        batch = db.batch()
        for job_id in job_ids:
            batch.update(db.collection(HUMAN_JOB_COLLECTION).document(job_id), {"payout_status": "paid", "workerPaymentStatus": "paid", "workerPaidAt": now, "updatedAt": firestore.SERVER_TIMESTAMP})
        await asyncio.to_thread(batch.commit)
    return {"status": "paid", "payout_id": payout_id, "jobs_marked_paid": len(items) or len(job_ids)}


@app.post("/api/admin/human-jobs/{job_id}/payment-deduction")
async def admin_deduct_worker_job_payment(job_id: str, request: Request):
    admin = _require_admin(request)
    if not db:
        raise HTTPException(status_code=503, detail="The workflow database is unavailable.")
    payload = await request.json()
    source = str(payload.get("source") or "job").strip().lower()
    segment_id = str(payload.get("segment_id") or "").strip() or None
    try:
        deduction_amount = int(payload.get("amount_kes") or 0)
    except (TypeError, ValueError):
        deduction_amount = 0
    reason = re.sub(r"\s+", " ", str(payload.get("reason") or "").strip())[:500]
    if source not in {"job", "segment", "proofreader"}:
        raise HTTPException(status_code=400, detail="Choose a valid worker payment item.")
    if source == "segment" and not segment_id:
        raise HTTPException(status_code=400, detail="Choose the submitted segment to adjust.")
    if deduction_amount <= 0 or not reason:
        raise HTTPException(status_code=400, detail="Enter a positive deduction and a reason for the record.")
    job_ref = db.collection(HUMAN_JOB_COLLECTION).document(job_id)
    job_snapshot = await asyncio.to_thread(job_ref.get)
    job = job_snapshot.to_dict() or {}
    archive_snapshot = None
    if job_snapshot.exists:
        matches = [item for item in _human_worker_earning_items(job_id, job, include_processed=True) if item.get("source") == source and (source != "segment" or item.get("segment_id") == segment_id)]
        item = matches[0] if matches else None
    else:
        archive_query = db.collection(HUMAN_EARNING_ARCHIVE_COLLECTION).where(filter=FieldFilter("job_id", "==", job_id))
        archive_snapshots = await asyncio.to_thread(lambda: list(archive_query.stream()))
        archive_snapshot = next((snap for snap in archive_snapshots if (snap.to_dict() or {}).get("source") == source and (source != "segment" or (snap.to_dict() or {}).get("segment_id") == segment_id)), None)
        item = archive_snapshot.to_dict() if archive_snapshot else None
    if not item:
        raise HTTPException(status_code=404, detail="That worker earning was not found.")
    if item.get("payout_status") == "paid" or item.get("paid_at"):
        raise HTTPException(status_code=409, detail="Already-paid earnings cannot be adjusted.")
    gross = int(item.get("gross_amount_kes") or item.get("amount_kes") or 0)
    previous_deduction = int(item.get("deduction_kes") or 0)
    new_deduction = previous_deduction + deduction_amount
    if new_deduction > gross:
        raise HTTPException(status_code=400, detail=f"The deduction cannot exceed the unpaid gross amount of KES {gross - previous_deduction}.")
    new_amount = gross - new_deduction
    payout_period_id = item.get("payout_period_id")
    payout_ref = None
    payout_update = None
    if payout_period_id:
        payout_ref = db.collection(HUMAN_PAYOUT_COLLECTION).document(f"{item.get('worker_uid')}_{payout_period_id}")
        payout_snapshot = await asyncio.to_thread(payout_ref.get)
        if payout_snapshot.exists:
            payout = payout_snapshot.to_dict() or {}
            if payout.get("status") == "paid":
                raise HTTPException(status_code=409, detail="Already-paid earnings cannot be adjusted.")
            invoice_items = [dict(x or {}) for x in (payout.get("items") or [])]
            invoice_item = next((x for x in invoice_items if x.get("job_id") == job_id and x.get("source") == source and (source != "segment" or x.get("segment_id") == segment_id)), None)
            if not invoice_item:
                raise HTTPException(status_code=409, detail="The payout invoice could not be reconciled, so no deduction was saved.")
            previous_amount = int(invoice_item.get("amount_kes") or 0)
            invoice_item.update({"amount_kes": new_amount, "gross_amount_kes": gross, "deduction_kes": new_deduction, "deduction_reason": reason})
            payout_update = {
                "items": invoice_items,
                "total_amount_kes": max(0, int(payout.get("total_amount_kes") or 0) - previous_amount + new_amount),
                "updatedAt": firestore.SERVER_TIMESTAMP,
            }
        elif item.get("payout_status") == "invoiced":
            raise HTTPException(status_code=409, detail="The pending payout invoice could not be found, so no deduction was saved.")
    batch = db.batch()
    if archive_snapshot is not None:
        archive_ref = archive_snapshot.reference
        batch.set(archive_ref, {"gross_amount_kes": gross, "deduction_kes": new_deduction, "deduction_reason": reason, "amount_kes": new_amount, "updated_at": datetime.now()}, merge=True)
    elif source == "job":
        batch.update(job_ref, {"worker_gross_amount_kes": gross, "worker_deduction_kes": new_deduction, "worker_deduction_reason": reason, "worker_amount_kes": new_amount, "updatedAt": firestore.SERVER_TIMESTAMP})
    elif source == "proofreader":
        batch.update(job_ref, {"proofreader_gross_amount_kes": gross, "proofreader_deduction_kes": new_deduction, "proofreader_deduction_reason": reason, "proofreader_amount_kes": new_amount, "updatedAt": firestore.SERVER_TIMESTAMP})
    else:
        segments = [dict(x or {}) for x in (job.get("segments") or [])]
        found_segment = False
        for segment in segments:
            if segment.get("id") == segment_id:
                segment.update({"worker_gross_amount_kes": gross, "worker_deduction_kes": new_deduction, "worker_deduction_reason": reason, "worker_amount_kes": new_amount})
                found_segment = True
        if not found_segment:
            raise HTTPException(status_code=404, detail="That worker segment was not found.")
        batch.update(job_ref, {"segments": segments, "updatedAt": firestore.SERVER_TIMESTAMP})
    if payout_ref is not None and payout_update is not None:
        batch.set(payout_ref, payout_update, merge=True)
    audit_ref = db.collection(HUMAN_DEDUCTION_COLLECTION).document(uuid.uuid4().hex)
    batch.set(audit_ref, {
        "job_id": job_id, "source": source, "segment_id": segment_id, "worker_uid": item.get("worker_uid"),
        "worker_email": item.get("worker_email") or "", "worker_name": item.get("worker_name") or "",
        "amount_kes": deduction_amount, "previous_deduction_kes": previous_deduction, "total_deduction_kes": new_deduction,
        "gross_amount_kes": gross, "remaining_amount_kes": new_amount, "reason": reason,
        "admin_email": (admin.get("email") or "").lower(), "created_at": datetime.now(),
    })
    await asyncio.to_thread(batch.commit)
    logger.info("Main admin %s deducted KES %s from %s earning for job %s: %s", (admin.get("email") or "").lower(), deduction_amount, source, job_id, reason)
    return {"job_id": job_id, "source": source, "deduction_kes": deduction_amount, "total_deduction_kes": new_deduction, "gross_amount_kes": gross, "amount_kes": new_amount, "reason": reason}


@app.post("/human-transcription/jobs/{job_id}/approve")
async def human_admin_approve(job_id: str, request: Request):
    _require_human_job_admin(request)
    job = await _human_job(job_id)
    if job.get("status") != "pending_admin":
        raise HTTPException(status_code=409, detail="This job is not waiting for admin approval.")
    await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {"status": "approved", "updatedAt": firestore.SERVER_TIMESTAMP})
    return {"status": "approved", "job_id": job_id}


@app.post("/human-transcription/jobs/{job_id}/assign")
async def human_admin_assign(job_id: str, request: Request):
    _require_human_job_admin(request)
    payload = await request.json()
    job = await _human_job(job_id)
    assignment_mode = str(payload.get("assignment_mode") or "single").strip().lower()
    requested_workers = payload.get("workers") if isinstance(payload.get("workers"), list) else []
    if not requested_workers and payload.get("worker_uid"):
        requested_workers = [{
            "worker_uid": payload.get("worker_uid"),
            "worker_email": payload.get("worker_email"),
            "worker_name": payload.get("worker_name"),
        }]
    requested_workers = [item for item in requested_workers if str(item.get("worker_uid") or "").strip()]
    if assignment_mode == "dual" or len(requested_workers) == 2:
        if len(requested_workers) != 2:
            raise HTTPException(status_code=400, detail="Choose a worker for each part of the split assignment.")
        if job.get("split_mode") == "dual" and job.get("segments"):
            raise HTTPException(status_code=409, detail="This job is already split. Assign or reassign one part at a time so submitted work is preserved.")
        worker_uids = [str(item.get("worker_uid")).strip() for item in requested_workers]
        same_worker_sequential = worker_uids[0] == worker_uids[1]
        if job.get("status") not in {"approved", "split_assigned", "split_in_progress", "proofreading_available", "proofreading_assigned"}:
            raise HTTPException(status_code=409, detail="Approve the job before assigning its parts.")
        try:
            total_seconds = float(job.get("seconds") or 0)
        except (TypeError, ValueError):
            total_seconds = 0
        midpoint = total_seconds / 2.0
        total_minutes = max(1, int(job.get("minutes") or math.ceil(total_seconds / 60.0)))
        first_minutes = max(1, int(math.ceil(total_minutes / 2.0)))
        second_minutes = max(1, total_minutes - first_minutes)
        part_seconds = [midpoint, max(1.0, total_seconds - midpoint)]
        now = datetime.now()
        segments = []
        for index, worker in enumerate(requested_workers):
            staged_part = same_worker_sequential and index == 1
            tat_seconds = human_tat_seconds(part_seconds[index]) if not staged_part else None
            segments.append({
                "id": f"part_{index + 1}",
                "label": f"Part {index + 1} of 2",
                "index": index + 1,
                "start_seconds": 0 if index == 0 else round(midpoint, 2),
                "end_seconds": round(midpoint, 2) if index == 0 else round(total_seconds, 2),
                "minutes": first_minutes if index == 0 else second_minutes,
                "worker_uid": None if staged_part else str(worker.get("worker_uid")).strip(),
                "worker_email": "" if staged_part else str(worker.get("worker_email") or "").strip().lower(),
                "worker_name": "" if staged_part else str(worker.get("worker_name") or "").strip(),
                "status": "available" if staged_part else "assigned",
                "assignedAt": None if staged_part else now,
                "deadlineAt": None if staged_part else now + timedelta(seconds=tat_seconds),
                "tat_seconds": tat_seconds,
                "tat_extension_minutes": 0,
                "transcript": "",
                "worker_notes": "",
                "final_attachment": None,
                "workerCompletedAt": None,
                "worker_minutes": None,
                "worker_amount_kes": None,
                "payout_status": None,
                "payout_period_id": None,
                "auto_reassigned_count": 0,
                "last_auto_reassigned_at": None,
                "last_auto_reassigned_worker_name": None,
            })
        updates = {
            "status": "split_assigned",
            "split_mode": "dual",
            "segments": segments,
            "assigned_worker_uids": list(dict.fromkeys(item.get("worker_uid") for item in segments if item.get("worker_uid"))),
            "worker_uid": None,
            "worker_email": None,
            "worker_name": None,
            "assignedAt": now,
            "deadlineAt": None,
            "tat_seconds": None,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
        await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, updates)
        return {
            "status": "split_assigned",
            "job_id": job_id,
            "same_worker_sequential": same_worker_sequential,
            "segments": [{"id": item["id"], "worker_uid": item["worker_uid"], "status": item["status"], "tat_seconds": item["tat_seconds"]} for item in segments],
        }

    if len(requested_workers) != 1:
        raise HTTPException(status_code=400, detail="Choose an approved worker first.")
    worker = requested_workers[0]
    worker_uid = str(worker.get("worker_uid") or "").strip()
    if not worker_uid:
        raise HTTPException(status_code=400, detail="Choose an approved worker first.")
    if job.get("split_mode") == "dual":
        segment_id = str(payload.get("segment_id") or "").strip()
        if not segment_id:
            raise HTTPException(status_code=400, detail="Choose a part waiting for assignment.")
        segments = [dict(item or {}) for item in (job.get("segments") or [])]
        target = next((item for item in segments if item.get("id") == segment_id), None)
        if not target or target.get("status") not in {"available", "approved"}:
            raise HTTPException(status_code=409, detail="That part is not waiting for assignment.")
        active_part = next((item for item in segments if item.get("id") != target.get("id") and item.get("worker_uid") == worker_uid and item.get("status") in {"assigned", "in_progress"}), None)
        if active_part:
            label = active_part.get("label") or "the current part"
            raise HTTPException(status_code=409, detail=f"Let this worker submit {label} before assigning another part to them.")
        tat_seconds = human_tat_seconds(float(target.get("end_seconds") or 0) - float(target.get("start_seconds") or 0))
        now = datetime.now()
        target.update({
            "worker_uid": worker_uid,
            "worker_email": str(worker.get("worker_email") or "").strip().lower(),
            "worker_name": str(worker.get("worker_name") or "").strip(),
            "status": "assigned",
            "assignedAt": now,
            "deadlineAt": now + timedelta(seconds=tat_seconds),
            "tat_seconds": tat_seconds,
        })
        await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {
            "segments": segments,
            "assigned_worker_uids": list(dict.fromkeys(item.get("worker_uid") for item in segments if item.get("worker_uid"))),
            "status": "split_assigned",
            "updatedAt": firestore.SERVER_TIMESTAMP,
        })
        return {"status": "split_assigned", "job_id": job_id, "segment_id": segment_id}

    if job.get("status") not in {"approved", "assigned"}:
        raise HTTPException(status_code=409, detail="Approve the job before assigning it.")
    tat_seconds = human_tat_seconds(float(job.get("seconds") or 0))
    now = datetime.now()
    deadline = now + timedelta(seconds=tat_seconds)
    updates = {
        "status": "assigned",
        "worker_uid": worker_uid,
        "worker_email": str(worker.get("worker_email") or "").strip().lower(),
        "worker_name": str(worker.get("worker_name") or "").strip(),
        "assignedAt": now,
        "deadlineAt": deadline,
        "tat_seconds": tat_seconds,
        "tat_extension_minutes": 0,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, updates)
    return {"status": "assigned", "job_id": job_id, "worker_uid": worker_uid, "tat_seconds": tat_seconds}


@app.post("/human-transcription/jobs/{job_id}/assign-proofreader")
async def human_admin_assign_proofreader(job_id: str, request: Request):
    _require_human_job_admin(request)
    payload = await request.json()
    worker_uid = str(payload.get("worker_uid") or "").strip()
    if not worker_uid:
        raise HTTPException(status_code=400, detail="Choose an approved proofreader first.")
    job = await _human_job(job_id)
    if job.get("split_mode") != "dual" or not all((item or {}).get("status") == "submitted" for item in (job.get("segments") or [])):
        raise HTTPException(status_code=409, detail="Both parts must be submitted before assigning the final proofreader.")
    tat_seconds = human_proofreading_tat_seconds(float(job.get("seconds") or 0))
    now = datetime.now()
    assigned = list(job.get("assigned_worker_uids") or [])
    if worker_uid not in assigned:
        assigned.append(worker_uid)
    await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {
        "status": "proofreading_assigned",
        "proofreader_uid": worker_uid,
        "proofreader_email": str(payload.get("worker_email") or "").strip().lower(),
        "proofreader_name": str(payload.get("worker_name") or "").strip(),
        "proofreader_status": "assigned",
        "proofreader_assignedAt": now,
        "proofreader_deadlineAt": now + timedelta(seconds=tat_seconds),
        "proofreader_tat_seconds": tat_seconds,
        "proofreader_tat_extension_minutes": 0,
        "assigned_worker_uids": assigned,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })
    return {"status": "proofreading_assigned", "job_id": job_id, "worker_uid": worker_uid, "tat_seconds": tat_seconds}


@app.post("/human-transcription/jobs/{job_id}/take-back")
async def human_admin_take_back(job_id: str, request: Request):
    """Return an active worker assignment to its queue without undoing submitted work."""
    admin = _require_human_job_admin(request)
    payload = await request.json()
    role = str(payload.get("role") or "transcriber").strip().lower()
    if role not in {"transcriber", "proofreader"}:
        raise HTTPException(status_code=400, detail="Choose a transcription part or proofreading assignment.")
    job = await _human_job(job_id)
    now = datetime.now()
    active_statuses = {"assigned", "in_progress"}
    worker_uid = ""
    worker_name = ""
    label = "Transcription"
    updates = {"updatedAt": firestore.SERVER_TIMESTAMP}

    if role == "proofreader":
        if job.get("proofreader_status") not in active_statuses or not job.get("proofreader_uid"):
            raise HTTPException(status_code=409, detail="Only an active proofreading assignment can be taken back.")
        worker_uid = str(job.get("proofreader_uid") or "")
        worker_name = str(job.get("proofreader_name") or job.get("proofreader_email") or "")
        label = "Final proofreading"
        segments = [dict(item or {}) for item in (job.get("segments") or [])]
        updates.update({
            "proofreader_status": "available", "proofreader_uid": None, "proofreader_email": None,
            "proofreader_name": None, "proofreader_assignedAt": None, "proofreader_deadlineAt": None,
            "proofreader_tat_seconds": None, "proofreader_tat_extension_minutes": 0,
            "assigned_worker_uids": list(dict.fromkeys(item.get("worker_uid") for item in segments if item.get("worker_uid"))),
            "status": "proofreading_available" if job.get("split_mode") == "dual" else "approved",
        })
    elif job.get("split_mode") == "dual":
        segment_id = str(payload.get("segment_id") or "").strip()
        if not segment_id:
            raise HTTPException(status_code=400, detail="Choose the active part to take back.")
        segments = [dict(item or {}) for item in (job.get("segments") or [])]
        target = next((item for item in segments if str(item.get("id") or "") == segment_id), None)
        if not target or target.get("status") not in active_statuses or not target.get("worker_uid"):
            raise HTTPException(status_code=409, detail="Only an active part can be taken back.")
        worker_uid = str(target.get("worker_uid") or "")
        worker_name = str(target.get("worker_name") or target.get("worker_email") or "")
        label = str(target.get("label") or "Transcription part")
        # Keep any draft text or attachments on the part. Only clear the live
        # assignment; previously submitted sibling parts are left untouched.
        target.update({
            "status": "available", "worker_uid": None, "worker_email": None, "worker_name": None,
            "assignedAt": None, "deadlineAt": None, "tat_seconds": None, "tat_extension_minutes": 0,
        })
        proofreader_uid = job.get("proofreader_uid") if job.get("proofreader_status") in active_statuses | {"submitted"} else None
        assigned_uids = [item.get("worker_uid") for item in segments if item.get("worker_uid")]
        if proofreader_uid:
            assigned_uids.append(proofreader_uid)
        if job.get("proofreader_status") in active_statuses:
            next_status = "proofreading_in_progress" if job.get("proofreader_status") == "in_progress" else "proofreading_assigned"
        elif all(item.get("status") == "submitted" for item in segments):
            next_status = "proofreading_available"
        elif any(item.get("status") == "in_progress" for item in segments):
            next_status = "split_in_progress"
        else:
            next_status = "split_assigned"
        updates.update({"segments": segments, "assigned_worker_uids": list(dict.fromkeys(assigned_uids)), "status": next_status})
    else:
        if job.get("status") not in active_statuses or not job.get("worker_uid"):
            raise HTTPException(status_code=409, detail="Only an active transcription assignment can be taken back.")
        worker_uid = str(job.get("worker_uid") or "")
        worker_name = str(job.get("worker_name") or job.get("worker_email") or "")
        updates.update({
            "status": "approved", "worker_uid": None, "worker_email": None, "worker_name": None,
            "assignedAt": None, "deadlineAt": None, "tat_seconds": None,
            "tat_extension_minutes": 0, "assigned_worker_uids": [],
        })

    updates.update({
        "last_assignment_takeback_at": now,
        "last_assignment_takeback_worker_uid": worker_uid,
        "last_assignment_takeback_worker_name": worker_name,
        "last_assignment_takeback_role": role,
        "last_assignment_takeback_label": label,
        "last_assignment_takeback_by_uid": str(admin.get("uid") or ""),
        "last_assignment_takeback_by_email": str(admin.get("email") or "").strip().lower(),
    })
    await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, updates)
    return {"status": updates.get("status"), "job_id": job_id, "role": role, "label": label, "worker_uid": worker_uid}


@app.post("/human-transcription/jobs/{job_id}/extend-tat")
async def human_admin_extend_tat(job_id: str, request: Request):
    _require_human_job_admin(request)
    payload = await request.json()
    try:
        extra_minutes = int(payload.get("minutes"))
    except (TypeError, ValueError):
        extra_minutes = 0
    if extra_minutes not in HUMAN_TAT_EXTENSION_OPTIONS:
        raise HTTPException(status_code=400, detail="Choose a 5, 10, 15, or 20 minute extension.")
    job = await _human_job(job_id)
    now = datetime.now()
    segment_id = str(payload.get("segment_id") or "").strip()
    if job.get("proofreader_uid") and str(payload.get("target") or "") == "proofreader":
        deadline = _as_dt(job.get("proofreader_deadlineAt"))
        if job.get("proofreader_status") not in {"assigned", "in_progress"} or not deadline or deadline <= now:
            raise HTTPException(status_code=409, detail="The proofreader deadline is no longer active.")
        await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {
            "proofreader_deadlineAt": deadline + timedelta(minutes=extra_minutes),
            "proofreader_tat_seconds": int(job.get("proofreader_tat_seconds") or 0) + extra_minutes * 60,
            "proofreader_tat_extension_minutes": int(job.get("proofreader_tat_extension_minutes") or 0) + extra_minutes,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        })
        return {"status": "extended", "job_id": job_id, "target": "proofreader", "minutes_added": extra_minutes}
    if job.get("split_mode") == "dual":
        segments = [dict(item or {}) for item in (job.get("segments") or [])]
        target = next((item for item in segments if item.get("id") == segment_id), None) if segment_id else None
        if not target:
            raise HTTPException(status_code=400, detail="Choose the part whose deadline should be extended.")
        if target.get("status") not in {"assigned", "in_progress"} or not _as_dt(target.get("deadlineAt")) or _as_dt(target.get("deadlineAt")) <= now:
            raise HTTPException(status_code=409, detail="That part is no longer active, so its deadline cannot be extended.")
        target["deadlineAt"] = _as_dt(target.get("deadlineAt")) + timedelta(minutes=extra_minutes)
        target["tat_seconds"] = int(target.get("tat_seconds") or 0) + extra_minutes * 60
        target["tat_extension_minutes"] = int(target.get("tat_extension_minutes") or 0) + extra_minutes
        await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {"segments": segments, "updatedAt": firestore.SERVER_TIMESTAMP})
        return {"status": "extended", "job_id": job_id, "segment_id": segment_id, "minutes_added": extra_minutes}
    deadline = _as_dt(job.get("deadlineAt"))
    if job.get("status") not in {"assigned", "in_progress"} or not deadline or deadline <= now:
        raise HTTPException(status_code=409, detail="This job's deadline is no longer active.")
    await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {
        "deadlineAt": deadline + timedelta(minutes=extra_minutes),
        "tat_seconds": int(job.get("tat_seconds") or 0) + extra_minutes * 60,
        "tat_extension_minutes": int(job.get("tat_extension_minutes") or 0) + extra_minutes,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })
    return {"status": "extended", "job_id": job_id, "minutes_added": extra_minutes}


@app.post("/human-transcription/jobs/{job_id}/start")
async def human_worker_start(job_id: str, request: Request):
    actor = await _human_actor(request)
    if actor["role"] != "worker":
        raise HTTPException(status_code=403, detail="Worker access is required.")
    job = await _human_job(job_id)
    await _human_assert_access(job, actor, allow_admin=False)
    if job.get("split_mode") == "dual":
        segments = [dict(item or {}) for item in (job.get("segments") or [])]
        target = next((item for item in segments if item.get("worker_uid") == actor["uid"] and item.get("status") in {"assigned", "in_progress"}), None)
        if target:
            target["status"] = "in_progress"
            await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {"segments": segments, "status": "split_in_progress", "updatedAt": firestore.SERVER_TIMESTAMP})
            return {"status": "in_progress", "job_id": job_id, "segment_id": target.get("id")}
        if job.get("proofreader_uid") == actor["uid"] and job.get("proofreader_status") in {"assigned", "in_progress"}:
            await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {"proofreader_status": "in_progress", "status": "proofreading_in_progress", "updatedAt": firestore.SERVER_TIMESTAMP})
            return {"status": "in_progress", "job_id": job_id, "role": "proofreader"}
        raise HTTPException(status_code=409, detail="This assignment is not ready to start.")
    if job.get("worker_uid") != actor["uid"]:
        raise HTTPException(status_code=403, detail="You do not have access to this job.")
    if job.get("status") not in {"assigned", "in_progress"}:
        raise HTTPException(status_code=409, detail="This job is not ready to start.")
    await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {"status": "in_progress", "updatedAt": firestore.SERVER_TIMESTAMP})
    return {"status": "in_progress", "job_id": job_id}


@app.post("/human-transcription/jobs/{job_id}/submit")
async def human_worker_submit(
    job_id: str,
    request: Request,
    transcript: str = Form(""),
    notes: str = Form(""),
    attachment: UploadFile = File(None),
):
    actor = await _human_actor(request)
    if actor["role"] != "worker":
        raise HTTPException(status_code=403, detail="Worker access is required.")
    job = await _human_job(job_id)
    await _human_assert_access(job, actor, allow_admin=False)
    transcript_text = str(transcript or "").strip()

    if job.get("split_mode") == "dual":
        segments = [dict(item or {}) for item in (job.get("segments") or [])]
        target = next((item for item in segments if item.get("worker_uid") == actor["uid"] and item.get("status") in {"assigned", "in_progress"}), None)
        if target:
            final_attachment = target.get("final_attachment")
            if attachment and attachment.filename:
                final_attachment = await _human_store_upload(job_id, attachment, f"final-{target.get('id')}")
            if not transcript_text and not final_attachment:
                raise HTTPException(status_code=400, detail="Add your part of the transcript, or attach the finished file, before submitting.")
            quote = job.get("quote") or {}
            minutes = int(target.get("minutes") or 0)
            rate = int(quote.get("transcriber_payout_kes_per_minute") or HUMAN_STANDARD_PAYOUT_KES)
            target.update({
                "status": "submitted",
                "transcript": transcript_text[:1000000],
                "final_attachment": final_attachment,
                "worker_notes": str(notes or "")[:12000],
                "submittedAt": datetime.now(),
                "workerCompletedAt": datetime.now(),
                "worker_minutes": minutes,
                "worker_amount_kes": max(0, minutes * rate),
                "payout_status": "unassigned",
            })
            both_submitted = all(item.get("status") == "submitted" for item in segments)
            await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {
                "segments": segments,
                "status": "proofreading_available" if both_submitted else "split_in_progress",
                "updatedAt": firestore.SERVER_TIMESTAMP,
            })
            return {"status": "proofreading_available" if both_submitted else "split_in_progress", "job_id": job_id, "segment_id": target.get("id")}

        if job.get("proofreader_uid") == actor["uid"] and job.get("proofreader_status") in {"assigned", "in_progress"}:
            final_attachment = job.get("final_attachment")
            if attachment and attachment.filename:
                final_attachment = await _human_store_upload(job_id, attachment, "final")
            combined = transcript_text or "\n\n".join(item.get("transcript", "") for item in segments if item.get("transcript"))
            if not combined and not final_attachment:
                raise HTTPException(status_code=400, detail="Review both parts and submit the combined transcript or a finished file.")
            minutes = int(job.get("minutes") or 0)
            await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {
                "status": "submitted",
                "transcript": combined[:1000000],
                "final_attachment": final_attachment,
                "worker_notes": str(notes or "")[:12000],
                "proofreader_status": "submitted",
                "proofreader_completedAt": datetime.now(),
                "proofreader_minutes": minutes,
                "proofreader_amount_kes": max(0, minutes * HUMAN_PROOFREADING_PAYOUT_KES),
                "proofreader_payout_status": "unassigned",
                "submittedAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            })
            return {"status": "submitted", "job_id": job_id, "role": "proofreader"}
        raise HTTPException(status_code=409, detail="This assignment is no longer active.")

    if job.get("worker_uid") != actor["uid"]:
        raise HTTPException(status_code=403, detail="You do not have access to this job.")
    transcript_text = str(transcript or "").strip()
    final_attachment = job.get("final_attachment")
    if attachment and attachment.filename:
        final_attachment = await _human_store_upload(job_id, attachment, "final")
    if not transcript_text and not final_attachment:
        raise HTTPException(status_code=400, detail="Add the completed transcript, or attach the finished file, before submitting.")
    quote = job.get("quote") or {}
    minutes = int(job.get("minutes") or quote.get("minutes") or 0)
    rate = int(quote.get("transcriber_payout_kes_per_minute") or HUMAN_STANDARD_PAYOUT_KES)
    worker_amount_kes = max(0, minutes * rate)
    updates = {
        "status": "submitted",
        "transcript": transcript_text[:1000000],
        "final_attachment": final_attachment,
        "worker_notes": str(notes or "")[:12000],
        "submittedAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "workerCompletedAt": datetime.now(),
        "worker_minutes": minutes,
        "worker_amount_kes": worker_amount_kes,
        "payout_status": "unassigned",
    }
    await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, updates)
    return {"status": "submitted", "job_id": job_id}


@app.post("/human-transcription/jobs/{job_id}/review")
async def human_admin_review(job_id: str, request: Request):
    _require_human_job_admin(request)
    job = await _human_job(job_id)
    if job.get("status") not in {"submitted", "client_review"}:
        raise HTTPException(status_code=409, detail="This job is not ready for admin review.")
    payload = await request.json()
    rating = payload.get("rating")
    try:
        rating = max(1, min(5, int(rating))) if rating is not None else None
    except (TypeError, ValueError):
        rating = None
    updates = {"status": "client_review", "admin_feedback": str(payload.get("feedback") or "")[:12000], "worker_rating": rating, "reviewedAt": firestore.SERVER_TIMESTAMP, "updatedAt": firestore.SERVER_TIMESTAMP}
    await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, updates)
    return {"status": "client_review", "job_id": job_id}


@app.post("/human-transcription/jobs/{job_id}/client-approve")
async def human_client_approve(job_id: str, request: Request):
    actor = await _human_actor(request)
    job = await _human_job(job_id)
    if actor["role"] == "admin":
        # Some clients are fully hands-off and trust an admin's review more
        # than they want to log in and click approve themselves. A real
        # admin, or the dedicated human-job-admin account, may sign off on
        # any client's behalf here.
        pass
    elif actor["role"] != "client":
        raise HTTPException(status_code=403, detail="Client access is required.")
    else:
        await _human_assert_access(job, actor, allow_admin=False)
    if job.get("status") != "client_review":
        raise HTTPException(status_code=409, detail="The job is not waiting for your approval.")
    await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, {"status": "client_approved", "clientApprovedAt": firestore.SERVER_TIMESTAMP, "clientApprovedBy": actor["email"], "updatedAt": firestore.SERVER_TIMESTAMP})
    return {"status": "client_approved", "job_id": job_id, "credits_deducted": 0}


@app.post("/human-transcription/jobs/{job_id}/release")
async def human_admin_release(job_id: str, request: Request):
    _require_human_job_admin(request)
    job = await _human_job(job_id)
    if job.get("status") != "client_approved":
        raise HTTPException(status_code=409, detail="Client approval is required before releasing the work.")
    client_email = job.get("client_email") or ""
    if human_job_credits_exempt(client_email):
        # This job's "client" is a real admin or the dedicated human-job-admin
        # account (info@typemywordz.ai). Human-transcription jobs never cost
        # that account credits, even though it still pays normally for AI
        # transcription and Ask TypeMyworDz elsewhere in the app.
        charge = {"charged": 0, "exempt": True}
    else:
        charge = await charge_credits(job.get("client_uid") or "", client_email, int(job.get("quote_credits") or 0), f"human transcription {job_id}")
        if charge.get("error") or charge.get("needed"):
            raise HTTPException(status_code=409, detail="The client's credits no longer cover this job. The work remains locked.")
    updates = {"status": "released", "credits_charged": int(charge.get("charged") or 0), "releasedAt": firestore.SERVER_TIMESTAMP, "updatedAt": firestore.SERVER_TIMESTAMP}
    await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).update, updates)
    return {"status": "released", "job_id": job_id, "credits_deducted": int(job.get("quote_credits") or 0)}


def _human_archive_id(job_id, source, segment_id=None):
    identity = f"{job_id}|{source}|{segment_id or 'main'}"
    return uuid.uuid5(uuid.NAMESPACE_URL, identity).hex


async def _human_archive_job_earnings(job_id, job, delete_source=True):
    """Copy every completed earning into a durable, transcript-free ledger."""
    earnings = list(_human_worker_earning_items(job_id, job, include_processed=True))
    if not earnings:
        if delete_source:
            await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).delete)
        return 0
    refs = []
    for item in earnings:
        archive_id = _human_archive_id(job_id, item["source"], item.get("segment_id"))
        ref = db.collection(HUMAN_EARNING_ARCHIVE_COLLECTION).document(archive_id)
        existing = await asyncio.to_thread(ref.get)
        existing_record = (existing.to_dict() or {}) if existing.exists else {}
        record = {
            "earning_id": archive_id, "job_id": job_id, "source": item["source"], "segment_id": item.get("segment_id"),
            "worker_uid": item.get("worker_uid"), "worker_email": item.get("worker_email") or "", "worker_name": item.get("worker_name") or "",
            "completed_at": item.get("completed_at"), "minutes": int(item.get("minutes") or 0),
            "gross_amount_kes": int(item.get("gross_amount_kes") or item.get("amount_kes") or 0),
            "deduction_kes": int(item.get("deduction_kes") or 0), "deduction_reason": item.get("deduction_reason") or "",
            "amount_kes": int(item.get("amount_kes") or 0), "payout_status": item.get("payout_status") or "unassigned",
            "payout_period_id": item.get("payout_period_id"), "paid_at": item.get("paid_at"),
            "job_status": job.get("status"), "archived_at": datetime.now(),
        }
        for key in ("payout_status", "payout_period_id", "paid_at"):
            if existing_record.get(key) is not None:
                record[key] = existing_record[key]
        refs.append((ref, record))
    if refs:
        batch = db.batch()
        for ref, record in refs:
            batch.set(ref, record, merge=True)
        if delete_source:
            batch.delete(db.collection(HUMAN_JOB_COLLECTION).document(job_id))
        await asyncio.to_thread(batch.commit)
    elif delete_source:
        await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).delete)
    return len(earnings)


async def _human_delete_job_storage(job_id, job, message_snapshots):
    bucket = _human_bucket()
    has_known_files = bool(job.get("audio") or job.get("audio_url") or job.get("audio_storage_path") or job.get("instruction_attachments") or job.get("final_attachment") or any((x or {}).get("final_attachment") for x in (job.get("segments") or [])) or any((x.to_dict() or {}).get("attachment") for x in message_snapshots))
    if bucket is None:
        if has_known_files:
            raise HTTPException(status_code=503, detail="File storage is unavailable, so this job was kept safely.")
        return 0
    try:
        prefix = f"human-workflow/{job_id}/"
        blobs = await asyncio.to_thread(lambda: list(bucket.list_blobs(prefix=prefix)))
        for blob in blobs:
            await asyncio.to_thread(blob.delete)
        return len(blobs)
    except Exception as exc:
        logger.error("Could not remove stored files for human job %s: %s", job_id, exc)
        raise HTTPException(status_code=503, detail="Some stored files could not be removed, so the job record was kept safely.")


async def _human_delete_job_messages(message_snapshots):
    for start in range(0, len(message_snapshots), 450):
        batch = db.batch()
        for snap in message_snapshots[start:start + 450]:
            batch.delete(snap.reference)
        await asyncio.to_thread(batch.commit)


async def _human_delete_job_safely(job_id, job):
    job_ref = db.collection(HUMAN_JOB_COLLECTION).document(job_id)
    message_snapshots = await asyncio.to_thread(lambda: list(job_ref.collection("messages").stream()))
    # Archive first, but keep the source job until files and messages are
    # removed. If any later cleanup step fails, a retry is safe and the worker
    # can still see the job; payment history is deduplicated by earning ID.
    earnings_archived = await _human_archive_job_earnings(job_id, job, delete_source=False)
    deleted_files = await _human_delete_job_storage(job_id, job, message_snapshots)
    await _human_delete_job_messages(message_snapshots)
    await asyncio.to_thread(job_ref.delete)
    return {"deleted": True, "job_id": job_id, "previous_status": job.get("status"), "files_deleted": deleted_files, "earnings_archived": earnings_archived}


@app.get("/api/admin/human-jobs/cleanup-candidates")
async def human_admin_cleanup_candidates(request: Request, start_date: str = "", end_date: str = ""):
    _require_admin(request)
    if not db:
        raise HTTPException(status_code=503, detail="The workflow database is unavailable.")
    try:
        start_bound = datetime.fromisoformat(start_date[:10]) if start_date else None
        end_bound = datetime.fromisoformat(end_date[:10]).replace(hour=23, minute=59, second=59, microsecond=999999) if end_date else None
    except Exception:
        raise HTTPException(status_code=400, detail="Use valid start and end dates.")
    if not start_bound or not end_bound or start_bound > end_bound:
        raise HTTPException(status_code=400, detail="Choose a valid date range before reviewing jobs for cleanup.")
    snapshots = await asyncio.to_thread(lambda: list(db.collection(HUMAN_JOB_COLLECTION).stream()))
    eligible = []
    for snap in snapshots:
        job = snap.to_dict() or {}
        if str(job.get("status") or "").lower() not in {"released", "cancelled"}:
            continue
        activity_date = _as_dt(job.get("releasedAt") or job.get("updatedAt") or job.get("createdAt"))
        if not activity_date or not (start_bound <= activity_date <= end_bound):
            continue
        earnings = list(_human_worker_earning_items(snap.id, job, include_processed=True))
        eligible.append({
            "job_id": snap.id, "job_number": job.get("job_number") or job.get("display_id") or snap.id,
            "status": job.get("status"), "created_at": _human_iso(_as_dt(job.get("createdAt"))),
            "cleanup_date": _human_iso(activity_date), "worker_count": len({x.get("worker_uid") for x in earnings if x.get("worker_uid")}),
            "earnings_preserved": len(earnings),
        })
    eligible.sort(key=lambda item: str(item.get("cleanup_date") or ""), reverse=True)
    return {"jobs": eligible, "count": len(eligible), "eligible_statuses": ["released", "cancelled"]}


@app.post("/api/admin/human-jobs/bulk-cleanup")
async def human_admin_bulk_cleanup(request: Request):
    _require_admin(request)
    if not db:
        raise HTTPException(status_code=503, detail="The workflow database is unavailable.")
    payload = await request.json()
    job_ids = payload.get("job_ids") if isinstance(payload.get("job_ids"), list) else []
    job_ids = list(dict.fromkeys(str(item).strip() for item in job_ids if str(item).strip()))
    if not job_ids or len(job_ids) > 500:
        raise HTTPException(status_code=400, detail="Select between 1 and 500 eligible jobs.")
    try:
        start_bound = datetime.fromisoformat(str(payload.get("start_date") or "")[:10])
        end_bound = datetime.fromisoformat(str(payload.get("end_date") or "")[:10]).replace(hour=23, minute=59, second=59, microsecond=999999)
    except Exception:
        raise HTTPException(status_code=400, detail="Choose the date range again before deleting.")
    if start_bound > end_bound:
        raise HTTPException(status_code=400, detail="The start date must be before the end date.")
    results, failures = [], []
    for job_id in job_ids:
        try:
            snapshot = await asyncio.to_thread(db.collection(HUMAN_JOB_COLLECTION).document(job_id).get)
            if not snapshot.exists:
                failures.append({"job_id": job_id, "error": "Job no longer exists."})
                continue
            job = snapshot.to_dict() or {}
            activity_date = _as_dt(job.get("releasedAt") or job.get("updatedAt") or job.get("createdAt"))
            if str(job.get("status") or "").lower() not in {"released", "cancelled"} or not activity_date or not (start_bound <= activity_date <= end_bound):
                failures.append({"job_id": job_id, "error": "Job is no longer eligible in the selected date range."})
                continue
            results.append(await _human_delete_job_safely(job_id, job))
        except HTTPException as exc:
            failures.append({"job_id": job_id, "error": exc.detail})
        except Exception as exc:
            logger.error("Could not clean up human job %s: %s", job_id, exc)
            failures.append({"job_id": job_id, "error": "Cleanup did not finish; the job should be reviewed."})
    return {"deleted": results, "failures": failures, "deleted_count": len(results), "failure_count": len(failures)}


@app.delete("/human-transcription/jobs/{job_id}")
async def human_admin_delete(job_id: str, request: Request):
    """Permanently remove a human/proofreading job while preserving earnings."""
    _require_human_job_admin(request)
    job = await _human_job(job_id)
    if not db:
        raise HTTPException(status_code=503, detail="Database is not ready.")
    try:
        return await _human_delete_job_safely(job_id, job)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Could not delete human job %s", job_id)
        raise HTTPException(status_code=500, detail="The job could not be fully removed. Worker payment history is protected; please retry, and contact support if the issue continues.")


def _human_thread_for(actor, requested_thread=""):
    """Which conversation is this request allowed to touch?

    A worker is never in contact with the client and a client is never in
    contact with the worker; every human job routes through admin instead.
    So there are two separate conversations per job, "client" and "worker",
    and admin is the only actor allowed to choose which one to open. A
    client or worker cannot pick a thread; the server picks it for them from
    their role, which is what actually keeps the two sides apart even if the
    browser were tricked into asking for the wrong one.
    """
    if actor["role"] == "client":
        return "client"
    if actor["role"] == "worker":
        return "worker"
    thread = (requested_thread or "").strip().lower()
    if thread not in ("client", "worker"):
        raise HTTPException(status_code=400, detail="Choose whether this message is to the client or the worker.")
    return thread


def _human_message_read_by_role(message, job):
    """Expose a role label for receipts without exposing participant UIDs."""
    message = message or {}
    job = job or {}
    sender_uid = str(message.get("sender_uid") or "")
    readers = {str(uid) for uid in (message.get("readBy") or []) if str(uid) and str(uid) != sender_uid}
    if not readers:
        return ""
    thread = str(message.get("thread") or "client").lower()
    worker_uids = {str(uid) for uid in (job.get("assigned_worker_uids") or []) if uid}
    worker_uids.update(str((item or {}).get("worker_uid")) for item in (job.get("segments") or []) if (item or {}).get("worker_uid"))
    for key in ("worker_uid", "proofreader_uid"):
        if job.get(key):
            worker_uids.add(str(job[key]))
    sender_role = str(message.get("sender_role") or "").lower()
    if not sender_role:
        if sender_uid == str(job.get("client_uid") or ""):
            sender_role = "client"
        elif sender_uid in worker_uids:
            sender_role = "worker"
        else:
            sender_role = "admin"
    if thread == "client":
        if sender_role == "admin":
            return "client" if str(job.get("client_uid") or "") in readers else ""
        return "admin"
    worker_readers = readers.intersection(worker_uids)
    if sender_role == "admin":
        return "worker" if worker_readers else ""
    return "worker" if worker_readers else "admin"


@app.get("/human-transcription/jobs/{job_id}/messages")
async def human_messages(job_id: str, request: Request, thread: str = ""):
    actor = await _human_actor(request)
    job = await _human_job(job_id)
    await _human_assert_access(job, actor)
    requested_thread = (thread or "").strip().lower()
    # The admin Messages inbox represents the whole job, not one of the two
    # internal job threads. When it opens a job without an explicit thread,
    # show the client thread by default and clear unread messages in both the
    # client and worker threads. The dedicated HumanJobWorkspace still passes
    # thread=client or thread=worker when the admin wants to read only one.
    inbox_open = actor["role"] == "admin" and not requested_thread
    target_thread = "client" if inbox_open else _human_thread_for(actor, requested_thread)
    threads_to_mark_read = {"client", "worker"} if inbox_open else {target_thread}
    ref = db.collection(HUMAN_JOB_COLLECTION).document(job_id).collection("messages")
    snapshots = await asyncio.to_thread(lambda: list(ref.order_by("createdAt").stream()))
    messages = []
    unread_refs = []
    for snap in snapshots:
        data = snap.to_dict() or {}
        # Messages saved before conversations were split have no thread on
        # them at all. Those were every one of them client<->admin, so that
        # is where they stay; they must never appear in a worker's thread.
        message_thread = data.get("thread") or "client"
        read_by = list(data.get("readBy") or [])
        if message_thread in threads_to_mark_read and data.get("sender_uid") != actor["uid"] and actor["uid"] not in read_by:
            unread_refs.append(snap.reference)
            read_by.append(actor["uid"])
            data["readBy"] = read_by
        if message_thread != target_thread:
            continue
        data["id"] = snap.id
        public_message = _human_public_for(data, actor["role"])
        public_message["read_by_role"] = _human_message_read_by_role(data, job)
        public_message.pop("readBy", None)
        messages.append(public_message)
    if unread_refs:
        batch = db.batch()
        for message_ref in unread_refs:
            batch.update(message_ref, {"readBy": firestore.ArrayUnion([actor["uid"]])})
        await asyncio.to_thread(batch.commit)
    return {"messages": messages, "thread": target_thread}


@app.post("/human-transcription/jobs/{job_id}/messages")
async def human_send_message(job_id: str, request: Request, attachment: UploadFile = File(None), message: str = Form(""), thread: str = Form("")):
    actor = await _human_actor(request)
    job = await _human_job(job_id)
    await _human_assert_access(job, actor)
    target_thread = _human_thread_for(actor, thread)
    if target_thread == "worker" and not job.get("worker_uid"):
        raise HTTPException(status_code=409, detail="This job has no worker assigned yet.")
    text = (message or "").strip()
    attachment_meta = await _human_store_upload(job_id, attachment, "chat") if attachment else None
    if not text and not attachment_meta:
        raise HTTPException(status_code=400, detail="Write a message or attach a file.")
    item = {"sender_uid": actor["uid"], "sender_email": actor["email"], "sender_role": actor["role"], "message": text[:12000], "attachment": attachment_meta, "thread": target_thread, "createdAt": firestore.SERVER_TIMESTAMP}
    job_ref = db.collection(HUMAN_JOB_COLLECTION).document(job_id)
    msg_ref = job_ref.collection("messages").document()
    message_summary = {
        "id": msg_ref.id,
        "sender_uid": actor["uid"],
        "sender_role": actor["role"],
        "thread": target_thread,
        "createdAt": firestore.SERVER_TIMESTAMP,
    }
    batch = db.batch()
    batch.set(msg_ref, item)
    batch.update(job_ref, {
        "last_message": message_summary,
        f"last_message_by_thread.{target_thread}": message_summary,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })
    await asyncio.to_thread(batch.commit)
    saved_snapshot = await asyncio.to_thread(msg_ref.get)
    saved_item = saved_snapshot.to_dict() or item
    saved_item["id"] = msg_ref.id
    public_message = _human_public_for(saved_item, actor["role"])
    public_message["read_by_role"] = ""
    public_message.pop("readBy", None)
    return {"message": public_message}


@app.get("/human-transcription/jobs/{job_id}/messages/{message_id}/attachment")
async def human_message_attachment(job_id: str, message_id: str, request: Request):
    actor = await _human_actor(request)
    job = await _human_job(job_id)
    await _human_assert_access(job, actor)
    message_snapshot = await asyncio.to_thread(
        db.collection(HUMAN_JOB_COLLECTION).document(job_id).collection("messages").document(message_id).get
    )
    if not message_snapshot.exists:
        raise HTTPException(status_code=404, detail="That attachment was not found.")
    message = message_snapshot.to_dict() or {}
    if actor["role"] in ("client", "worker") and (message.get("thread") or "client") != actor["role"]:
        raise HTTPException(status_code=403, detail="You do not have access to this attachment.")
    meta = message.get("attachment") or {}
    path = meta.get("storage_path")
    bucket = _human_bucket()
    if not path or bucket is None:
        raise HTTPException(status_code=404, detail="That attachment is not available.")
    if not str(path).startswith(f"human-workflow/{job_id}/chat/"):
        raise HTTPException(status_code=403, detail="That attachment does not belong to this job.")
    blob = bucket.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="That attachment is no longer available.")
    raw = await asyncio.to_thread(blob.download_as_bytes)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(meta.get("name") or "attachment")) or "attachment"
    return Response(
        content=raw,
        media_type=meta.get("content_type") or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/human-transcription/jobs/{job_id}/instruction/{attachment_index}")
async def human_instruction_attachment(job_id: str, attachment_index: int, request: Request):
    actor = await _human_actor(request)
    job = await _human_job(job_id)
    await _human_assert_access(job, actor)
    attachments = job.get("instruction_attachments") or []
    if attachment_index < 0 or attachment_index >= len(attachments):
        raise HTTPException(status_code=404, detail="That reference file was not found.")
    meta = attachments[attachment_index] or {}
    path = meta.get("storage_path")
    bucket = _human_bucket()
    if not path or bucket is None:
        raise HTTPException(status_code=404, detail="That reference file is not available.")
    if not str(path).startswith(f"human-workflow/{job_id}/instructions/"):
        raise HTTPException(status_code=403, detail="That reference file does not belong to this job.")
    blob = bucket.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="That reference file is no longer available.")
    raw = await asyncio.to_thread(blob.download_as_bytes)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(meta.get("name") or "reference-file")) or "reference-file"
    return Response(
        content=raw,
        media_type=meta.get("content_type") or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _human_segment_for_download(job, actor, segment_id):
    segment = next((dict(item or {}) for item in (job.get("segments") or []) if item.get("id") == segment_id), None)
    if not segment:
        raise HTTPException(status_code=404, detail="That finished part was not found.")
    is_assigned_proofreader = actor.get("role") == "worker" and job.get("proofreader_uid") == actor.get("uid") and job.get("proofreader_status") in {"assigned", "in_progress", "submitted"}
    is_own_segment = actor.get("role") == "worker" and segment.get("worker_uid") == actor.get("uid")
    if actor.get("role") != "admin" and not is_assigned_proofreader and not is_own_segment:
        raise HTTPException(status_code=403, detail="You do not have access to this finished part.")
    if segment.get("status") != "submitted":
        raise HTTPException(status_code=409, detail="This part has not been submitted yet.")
    return segment


@app.get("/human-transcription/jobs/{job_id}/segments/{segment_id}/transcript-download")
async def human_segment_transcript_download(job_id: str, segment_id: str, request: Request):
    actor = await _human_actor(request)
    job = await _human_job(job_id)
    segment = _human_segment_for_download(job, actor, segment_id)
    transcript = str(segment.get("transcript") or "").strip()
    if not transcript:
        raise HTTPException(status_code=404, detail="This part was submitted as an attached file, without editor text.")
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(segment.get("label") or segment_id))
    return Response(
        content=transcript,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_label}-transcript.txt"'},
    )


@app.get("/human-transcription/jobs/{job_id}/segments/{segment_id}/attachment")
async def human_segment_attachment_download(job_id: str, segment_id: str, request: Request):
    actor = await _human_actor(request)
    job = await _human_job(job_id)
    segment = _human_segment_for_download(job, actor, segment_id)
    meta = segment.get("final_attachment") or {}
    path = meta.get("storage_path")
    bucket = _human_bucket()
    if not path or bucket is None:
        raise HTTPException(status_code=404, detail="No finished file was attached to this part.")
    if not str(path).startswith(f"human-workflow/{job_id}/final-{segment_id}/"):
        raise HTTPException(status_code=403, detail="That finished file does not belong to this part.")
    blob = bucket.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="That finished file is no longer available.")
    raw = await asyncio.to_thread(blob.download_as_bytes)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(meta.get("name") or f"{segment_id}-finished-file")) or f"{segment_id}-finished-file"
    return Response(
        content=raw,
        media_type=meta.get("content_type") or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/human-transcription/jobs/{job_id}/final-attachment")
async def human_final_attachment(job_id: str, request: Request):
    actor = await _human_actor(request)
    job = await _human_job(job_id)
    await _human_assert_access(job, actor)
    meta = job.get("final_attachment") or {}
    path = meta.get("storage_path")
    bucket = _human_bucket()
    if not path or bucket is None:
        raise HTTPException(status_code=404, detail="No finished file was attached to this job.")
    if not str(path).startswith(f"human-workflow/{job_id}/final/"):
        raise HTTPException(status_code=403, detail="That file does not belong to this job.")
    if job.get("status") not in {"submitted", "client_review", "client_approved", "released"} and actor["role"] == "client":
        raise HTTPException(status_code=403, detail="The completed work will be downloadable after admin releases it.")
    blob = bucket.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="That file is no longer available.")
    raw = await asyncio.to_thread(blob.download_as_bytes)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(meta.get("name") or "final-transcript")) or "final-transcript"
    return Response(
        content=raw,
        media_type=meta.get("content_type") or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _human_worker_split_mp3_bytes(clipped):
    """Make a small speech-first MP3 without collapsing distinct speaker channels."""
    channels = clipped.split_to_mono()
    identical_channels = len(channels) > 1 and all(
        channel.raw_data == channels[0].raw_data for channel in channels[1:]
    )
    if len(channels) == 1 or identical_channels:
        # 22.05 kHz / 32 kbps mono keeps speech clear while halving the
        # size of the previous 64 kbps worker clip.
        speech_audio = channels[0].set_frame_rate(22050)
        bitrate = "32k"
    else:
        # Separate stereo tracks can contain different speakers. Preserve
        # those channels rather than downmixing them into one another.
        speech_audio = clipped.set_frame_rate(22050)
        bitrate = "64k"
    output = BytesIO()
    speech_audio.export(output, format="mp3", bitrate=bitrate)
    return output.getvalue()


@app.get("/human-transcription/jobs/{job_id}/audio")
async def human_audio(job_id: str, request: Request, segment_id: str = ""): 
    actor = await _human_actor(request)
    job = await _human_job(job_id)
    await _human_assert_access(job, actor)
    meta = job.get("audio") or {}
    path = meta.get("storage_path")
    bucket = _human_bucket()
    if not path or bucket is None:
        raise HTTPException(status_code=404, detail="The source audio is not available.")
    blob = bucket.blob(path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="The source audio is no longer available.")
    raw = await asyncio.to_thread(blob.download_as_bytes)
    # Split workers receive only their assigned half of the recording. Admins,
    # clients and proofreaders continue to receive the complete source audio.
    if segment_id and actor.get("role") == "worker" and job.get("split_mode") == "dual":
        segment = next((item for item in (job.get("segments") or []) if item.get("id") == segment_id and item.get("worker_uid") == actor.get("uid")), None)
        if not segment:
            raise HTTPException(status_code=403, detail="That audio segment is not assigned to you.")
        try:
            suffix = str(meta.get("name") or "").rsplit(".", 1)[-1].lower()
            source = AudioSegment.from_file(BytesIO(raw), format=suffix if suffix in {"mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac", "aac"} else None)
            start_ms = max(0, int(float(segment.get("start_seconds") or 0) * 1000))
            end_ms = min(len(source), int(float(segment.get("end_seconds") or len(source) / 1000) * 1000))
            clipped = source[start_ms:end_ms]
            raw = _human_worker_split_mp3_bytes(clipped)
            return Response(content=raw, media_type="audio/mpeg", headers={"Content-Disposition": f"inline; filename={segment.get('id') or 'assigned-part'}.mp3"})
        except Exception as exc:
            logger.warning("Could not clip split audio %s/%s: %s", job_id, segment_id, exc)
            # Never send the full source recording when a worker is authorized
            # for only one part and segment preparation fails.
            raise HTTPException(status_code=502, detail="The assigned audio segment could not be prepared. Please try again or contact the admin.")
    return Response(content=raw, media_type=meta.get("content_type") or "application/octet-stream", headers={"Content-Disposition": f"inline; filename={meta.get('name') or 'source-audio'}"})


@app.get("/human-transcription/jobs/{job_id}/download")
async def human_download(job_id: str, request: Request):
    actor = await _human_actor(request)
    job = await _human_job(job_id)
    await _human_assert_access(job, actor)
    if job.get("status") != "released" and actor["role"] == "client":
        raise HTTPException(status_code=403, detail="The completed work will be downloadable after admin releases it.")
    transcript = str(job.get("transcript") or "")
    if not transcript:
        raise HTTPException(status_code=404, detail="No completed transcript is available.")
    return Response(content=transcript, media_type="text/plain", headers={"Content-Disposition": f"attachment; filename=human-{job_id}.txt"})

TRAINING_GUIDELINES = {
    "title": "TypeMyworDz human-work guidelines",
    "summary": "TypeMyworDz trainees learn to produce accurate, faithful, readable transcripts. These standards apply to training exercises and to future human-work assignments unless a client brief says otherwise.",
    "sections": [
        {"title": "1. The TypeMyworDz standard", "body": "A transcript is a written record of spoken audio, not a summary or a rewrite. Preserve the speaker's meaning, wording, order, tone, and uncertainty. Do not silently paraphrase, repair grammar, add explanations, or invent words. Accuracy comes before speed, and every submission must receive a complete review before delivery."},
        {"title": "2. Full verbatim and clean verbatim", "body": "Full verbatim includes speech errors, false starts, fillers, slang, stutters, repetitions, and unfinished wording. Clean verbatim removes clear fillers, stutters, speech errors, false starts, unnecessary repetitions, and explicit self-corrections while preserving meaning. Do not change an intact sentence simply because it sounds informal. Follow the client brief when it specifies the style."},
        {"title": "3. Do not paraphrase or correct the speaker", "body": "Do not reconstruct a sentence into better English, change a speaker's grammar, or replace spoken wording with a more professional alternative unless the applicable client brief expressly requires it. Preserve contractions as spoken. Curse words are transcribed as spoken. Direct quotations use straight double quotation marks."},
        {"title": "4. Inaudible, unintelligible, and timestamps", "body": "Use [inaudible 00:00:00] when the sound cannot be heard. Use [unintelligible 00:00:00] when speech can be heard but cannot be understood. Use timestamps in [00:00:00] format. Do not use parentheses in place of square brackets. Check the audio before marking a passage and place the notation as close as possible to the uncertain words."},
        {"title": "5. Paragraphs, punctuation, and capitalization", "body": "Split long speeches into readable paragraphs without changing the speaker's order. Normally end every sentence with punctuation and capitalize the beginning of every sentence. Do not use exclamation marks in transcripts. Preserve question marks when the speaker asks a question. Never add punctuation that changes the intended meaning."},
        {"title": "6. Speaker labels and changes", "body": "Use descriptive speaker labels when the assignment requires labels. Labels should be bold, followed by a colon and one space, for example, Interviewer: and Participant: . Separate speaker changes as accurately as possible. When identity is uncertain, replay the exchange and flag the uncertainty instead of guessing."},
        {"title": "7. Sound events and interruptions", "body": "Use concise, lower-case bracketed notes for relevant sound events, such as [laughs], [background noise], [crosstalk], [silence], and [sound cut]. Use a double dash for a false start, speech error, or unfinished sentence. Use a single dash for an interruption where the speaker continues. Do not over-describe ordinary room sounds."},
        {"title": "8. Numbers, dates, and times", "body": "Generally spell out single-digit numbers and use numerals for larger numbers. Use numerals for money, years, ages, percentages, measurements, equations, dates, times, telephone numbers, and mixed-number sentences when the context calls for them. Write percent in transcript text unless a client brief says otherwise. Use capitalized AM and PM for times. Formal series remain capitalized, such as Grade 8, Section B, Chapter 1, and Article VI."},
        {"title": "9. Names, research, and consistency", "body": "Research distinctive proper nouns, organisations, places, technical terms, and titles when appropriate. Research verifies spelling and context; it does not authorize changing the speaker's wording or adding information. Keep confirmed spellings consistent throughout the transcript and ask the admin when two possible identities cannot be resolved."},
        {"title": "10. Privacy, review, and delivery", "body": "Treat every recording, transcript, name, and client instruction as confidential. Use approved tools and do not share files casually. Before delivery, check the brief, completeness, speaker turns, timestamps, uncertain passages, names, numbers, punctuation, and formatting from beginning to end. Submit work only when another person can use it without needing to reconstruct what you meant."},
    ],
}

TRAINING_LEVELS = [
    {"level": 1, "name": "Orientation and standards", "kind": "study", "description": "Read the programme, understand the workflow, and begin the first conversation with the admin."},
    {"level": 2, "name": "TypeMyworDz guidelines", "kind": "study", "description": "Study the human-work guidelines, ask questions, and discuss examples with the admin.", "guidelines": True},
    {"level": 3, "name": "Tools, privacy and review habits", "kind": "study", "description": "Learn the editor, timestamp checks, file handling, privacy expectations, and quality-control routine."},
    {"level": 4, "name": "Practical: clean transcript", "kind": "practical", "description": "Complete a short clean-transcript exercise using the required brief and formatting rules."},
    {"level": 5, "name": "Practical: speakers and timestamps", "kind": "practical", "description": "Complete a timestamp and speaker-review exercise, checking difficult audio carefully."},
    {"level": 6, "name": "Practical: client-ready delivery", "kind": "practical", "description": "Complete a full client-ready exercise and respond to admin feedback."},
]


async def _trainee_actor(request: Request):
    decoded = _verified_user(request)
    uid = decoded.get("uid") or ""
    email = (decoded.get("email") or "").strip().lower()
    profile = await _load_profile(uid) or {}
    return {"uid": uid, "email": email, "profile": profile}



@app.post("/api/trainee/complete-signup")
async def complete_trainee_signup(request: Request):
    actor = await _trainee_actor(request)
    payload = await request.json()
    reference = str(payload.get("reference") or "").strip()
    official_name = str(payload.get("official_name") or "").strip()
    if not reference or len(official_name) < 2:
        raise HTTPException(status_code=400, detail="The paid enrollment and official name are required.")
    if not db:
        raise HTTPException(status_code=503, detail="The enrollment database is unavailable.")
    snap = await asyncio.to_thread(db.collection("payment_intents").document(reference).get)
    intent = snap.to_dict() if snap.exists else {}
    if intent.get("product") != TRAINEE_PRODUCT:
        raise HTTPException(status_code=409, detail="This payment is not a trainee enrollment.")

    # A browser can reach account creation before the provider webhook has
    # updated Firestore. Re-check the provider here instead of creating a
    # normal client account and leaving the paid trainee half-enrolled.
    if intent.get("status") != "paid":
        provider = str(intent.get("provider") or "paystack").strip().lower()
        if provider == "kora":
            await verify_kora_and_enroll(reference)
        elif provider == "paystack":
            verification = await verify_paystack_payment(reference)
            if verification.get("status") == "success":
                await asyncio.to_thread(db.collection("payment_intents").document(reference).set, {
                    "status": "paid",
                    "paidAt": firestore.SERVER_TIMESTAMP,
                    "email": verification.get("email") or intent.get("email"),
                    "product": TRAINEE_PRODUCT,
                    "provider": "paystack",
                    "countryCode": intent.get("countryCode") or TRAINEE_COUNTRY,
                    "currency": verification.get("currency") or intent.get("currency") or "KES",
                }, merge=True)
        snap = await asyncio.to_thread(db.collection("payment_intents").document(reference).get)
        intent = snap.to_dict() if snap.exists else {}

    if intent.get("status") != "paid":
        raise HTTPException(status_code=409, detail="Payment has not been confirmed for this enrollment yet. Please wait a moment and try again.")
    if str(intent.get("email") or "").strip().lower() != actor["email"]:
        raise HTTPException(status_code=403, detail="This payment belongs to a different email address.")
    saved_official_name = str((actor.get("profile") or {}).get("officialIdName") or "").strip()
    if saved_official_name and saved_official_name.casefold() != official_name.casefold():
        raise HTTPException(status_code=409, detail="The official name from your trainee registration is locked. Contact support if it needs correcting.")
    result = await enroll_paid_trainee(actor["email"], reference, TRAINEE_PRICE_USD, str(intent.get("currency") or "KES"), TRAINEE_COUNTRY, str(intent.get("provider") or "paystack"), user_id=actor["uid"])
    if not result.get("success"):
        raise HTTPException(status_code=409, detail=result.get("error") or "The trainee account could not be completed.")
    await asyncio.to_thread(db.collection("users").document(actor["uid"]).set, {"name": official_name, "officialIdName": official_name}, merge=True)

    # Trainees bypass the normal profile-creation path, so send their own
    # welcome message after payment and enrolment have both succeeded. Email
    # delivery is deliberately non-blocking for account completion.
    trainee_subject, trainee_html, trainee_text = build_trainee_welcome_email(official_name)
    await _send_resend_message(actor["email"], trainee_subject, trainee_html, trainee_text, "Trainee welcome email")
    return {"success": True, "training_room": True}

@app.get("/human-transcription/trainee/status")
async def trainee_status(request: Request):
    actor = await _trainee_actor(request)
    profile = actor["profile"]
    return {
        "application": {
            "status": profile.get("traineeStatus") or "not_started",
            "payment_status": profile.get("trainingPaymentStatus") or "not_submitted",
            "name": profile.get("name") or profile.get("full_name") or "",
            "country": profile.get("country") or "",
            "notes": profile.get("traineeNotes") or "",
            "submitted_at": _human_iso(profile.get("traineeAppliedAt")),
        },
        "training": {
            "level": int(profile.get("trainingLevel") or 0),
            "status": profile.get("trainingStatus") or "not_started",
            "submissions": profile.get("trainingSubmissions") or {},
        },
        "levels": TRAINING_LEVELS,
        "guidelines": TRAINING_GUIDELINES,
        "is_worker": bool(profile.get("workerApproved") or str(profile.get("role") or "").lower() == "worker"),
    }


@app.post("/human-transcription/trainee/apply")
async def trainee_apply(request: Request):
    actor = await _trainee_actor(request)
    profile = actor["profile"]
    payload = await request.json()
    name = str(payload.get("name") or "").strip()
    country = str(payload.get("country") or "").strip()
    notes = str(payload.get("notes") or "").strip()
    if len(name) < 2 or country.upper() != "KE":
        raise HTTPException(status_code=400, detail="Enter your official ID name. Training enrollment is currently limited to Kenya.")
    registered_name = str((profile or {}).get("officialIdName") or "").strip()
    if registered_name and registered_name.casefold() != name.casefold():
        raise HTTPException(status_code=409, detail="Your official name is locked from trainee registration. Contact support if it needs correcting.")
    updates = {
        "name": name,
        "officialIdName": name,
        "country": "Kenya",
        "countryCode": "KE",
        "traineeNotes": notes[:12000],
        "traineeStatus": profile.get("traineeStatus") or "payment_pending",
        "trainingPaymentStatus": profile.get("trainingPaymentStatus") or "pending",
        "trainingRoomAccess": bool(profile.get("trainingRoomAccess")),
        "traineeAppliedAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }
    if not db:
        raise HTTPException(status_code=503, detail="The workflow database is unavailable.")
    await asyncio.to_thread(db.collection("users").document(actor["uid"]).set, updates, merge=True)
    return {"status": updates["traineeStatus"], "payment_status": updates["trainingPaymentStatus"], "training_room": bool(updates["trainingRoomAccess"])}


@app.get("/api/admin/trainees")
async def admin_trainees(request: Request):
    _require_admin(request)
    if not db:
        return {"trainees": []}
    rows = []
    for snap in await asyncio.to_thread(lambda: list(db.collection("users").stream())):
        data = snap.to_dict() or {}
        status = str(data.get("traineeStatus") or "").lower()
        role = str(data.get("role") or data.get("user_type") or "").lower()
        if not status and role not in {"trainee", "worker"}:
            continue
        data["uid"] = data.get("uid") or snap.id
        data["id"] = snap.id
        data["email"] = data.get("email") or ""
        data["name"] = data.get("name") or data.get("full_name") or data.get("displayName") or "Unnamed applicant"
        rows.append(_human_public(data))
    rows.sort(key=lambda item: str(item.get("traineeAppliedAt") or ""), reverse=True)
    return {"trainees": rows}


@app.post("/api/admin/trainees/{uid}/decision")
async def admin_trainee_decision(uid: str, request: Request):
    _require_admin(request)
    payload = await request.json()
    decision = str(payload.get("decision") or "").strip().lower()
    payment_status = str(payload.get("payment_status") or "").strip().lower()
    if decision not in {"reject", "promote_worker", "approve_level"}:
        raise HTTPException(status_code=400, detail="That trainee decision is not supported.")
    profile = await _load_profile(uid)
    if profile is None:
        raise HTTPException(status_code=404, detail="That trainee account was not found.")
    updates = {"updatedAt": firestore.SERVER_TIMESTAMP}
    if payment_status in {"verified", "rejected", "pending_verification"}:
        updates["trainingPaymentStatus"] = payment_status
    if decision == "reject":
        updates.update({"role": "client", "traineeStatus": "rejected", "trainingStatus": "rejected", "workerApproved": False})
    elif decision == "approve_level":
        current = max(1, int(profile.get("trainingLevel") or 1))
        updates.update({"role": "trainee", "traineeStatus": "enrolled", "trainingStatus": "active", "trainingRoomAccess": True, "trainingLevel": min(len(TRAINING_LEVELS), current + 1), "workerApproved": False})
    elif decision == "promote_worker":
        updates.update({"role": "worker", "traineeStatus": "enrolled", "trainingStatus": "completed", "trainingRoomAccess": False, "workerApproved": True, "workerApprovedAt": firestore.SERVER_TIMESTAMP})
    await asyncio.to_thread(db.collection("users").document(uid).set, updates, merge=True)
    return {"status": "updated", "uid": uid, "decision": decision}


@app.post("/human-transcription/trainee/training/{level}/submit")
async def trainee_submit_training(level: int, request: Request):
    actor = await _trainee_actor(request)
    profile = actor["profile"]
    role = str(profile.get("role") or profile.get("user_type") or "").lower()
    if role not in {"trainee", "worker"} and not profile.get("workerApproved"):
        raise HTTPException(status_code=403, detail="Trainee access is required.")
    if level < 1 or level > len(TRAINING_LEVELS):
        raise HTTPException(status_code=400, detail="That training level does not exist.")
    current_level = max(1, int(profile.get("trainingLevel") or 1))
    if level > current_level:
        raise HTTPException(status_code=403, detail="Complete the current module and wait for admin approval before opening that module.")
    payload = await request.json()
    transcript = str(payload.get("transcript") or "").strip()
    notes = str(payload.get("notes") or "").strip()
    if level >= 4 and not transcript:
        raise HTTPException(status_code=400, detail="Submit the completed practical transcript first.")
    submission = {"level": level, "transcript": transcript[:1000000], "notes": notes[:12000], "status": "submitted", "createdAt": firestore.SERVER_TIMESTAMP, "uid": actor["uid"], "email": actor["email"]}
    await asyncio.to_thread(db.collection("training_submissions").document(f"{actor['uid']}-{level}").set, submission, merge=True)
    updates = {"trainingSubmissions": {**(profile.get("trainingSubmissions") or {}), str(level): "submitted"}, "trainingStatus": "review", "updatedAt": firestore.SERVER_TIMESTAMP}
    await asyncio.to_thread(db.collection("users").document(actor["uid"]).set, updates, merge=True)
    return {"status": "submitted", "level": level}



@app.get("/api/messaging/contacts")
async def messaging_contacts(request: Request):
    """Return safe direct-message recipients for the signed-in dashboard."""
    actor = await _user_chat_actor(request)
    if not db:
        raise HTTPException(status_code=503, detail="Messaging is not available yet.")
    rows = []
    if actor["role"] == "admin":
        rows = await asyncio.to_thread(_read_admin_users_snapshot)
    else:
        for email in ADMIN_EMAILS:
            try:
                record = await asyncio.to_thread(firebase_auth.get_user_by_email, email)
                profile = await _load_profile(record.uid) or {}
                rows.append({
                    "uid": record.uid,
                    "email": email,
                    "name": profile.get("name") or profile.get("displayName") or email,
                    "role": "admin",
                })
            except Exception:
                continue
    return {"contacts": [item for item in rows if item.get("uid") != actor["uid"]]}


# ===================== Direct user conversations ============================
# A job conversation belongs to a human-work job. This separate collection is
# for a conversation between two accounts, so admins can contact a client,
# worker, trainee, or another admin without inventing a job just to send a note.
def _user_chat_thread_id(first_uid: str, second_uid: str) -> str:
    return "--".join(sorted([str(first_uid), str(second_uid)]))


async def _user_chat_target(other_uid: str):
    other_uid = str(other_uid or "").strip()
    if not other_uid:
        raise HTTPException(status_code=400, detail="A recipient is required.")
    snapshot = await asyncio.to_thread(db.collection("users").document(other_uid).get)
    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="That user could not be found.")
    data = snapshot.to_dict() or {}
    return {
        "uid": other_uid,
        "email": data.get("email") or "",
        "name": data.get("name") or data.get("full_name") or data.get("displayName") or data.get("email") or "User",
        "role": data.get("role") or data.get("user_type") or "client",
    }


async def _user_chat_actor(request: Request):
    decoded = _verified_user(request)
    uid = decoded.get("uid") or ""
    email = (decoded.get("email") or "").strip().lower()
    if not uid:
        raise HTTPException(status_code=401, detail="Your account could not be verified.")
    return {"uid": uid, "email": email, "role": "admin" if is_admin_user(email) else "user"}


async def _user_chat_file(thread_id: str, upload: UploadFile):
    if not upload or not upload.filename:
        return None
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The attached file is empty.")
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Attachments must be 25 MB or smaller.")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(upload.filename))[:180] or "attachment"
    path = f"user-chats/{thread_id}/{uuid.uuid4().hex}-{safe_name}"
    bucket = _human_bucket()
    if bucket is None:
        raise HTTPException(status_code=503, detail="File storage is not ready yet. Please try again shortly.")
    blob = bucket.blob(path)
    blob.upload_from_string(raw, content_type=upload.content_type or "application/octet-stream")
    return {"name": upload.filename, "storage_path": path, "content_type": upload.content_type or "application/octet-stream", "size": len(raw)}


@app.get("/api/user-chats/{other_uid}/messages")
async def user_chat_messages(other_uid: str, request: Request):
    actor = await _user_chat_actor(request)
    target = await _user_chat_target(other_uid)
    if target["uid"] == actor["uid"]:
        raise HTTPException(status_code=400, detail="You cannot start a conversation with yourself.")
    thread_id = _user_chat_thread_id(actor["uid"], target["uid"])
    thread_ref = db.collection("user_chats").document(thread_id)
    # Repair older conversations whose messages were written before the
    # parent-thread materialization fix.
    await asyncio.to_thread(thread_ref.set, {
        "participants": [actor["uid"], target["uid"]],
        "participant_emails": [actor["email"], target["email"]],
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }, merge=True)
    ref = thread_ref.collection("messages")
    snapshots = await asyncio.to_thread(lambda: list(ref.order_by("createdAt").stream()))
    unread_refs = []
    messages = []
    for snap in snapshots:
        data = snap.to_dict() or {}
        if data.get("recipient_uid") == actor["uid"] and not data.get("readAt"):
            unread_refs.append(snap.reference)
        data["id"] = snap.id
        messages.append(_human_public(data))
    if unread_refs:
        batch = db.batch()
        for message_ref in unread_refs:
            batch.update(message_ref, {"readAt": firestore.SERVER_TIMESTAMP})
        await asyncio.to_thread(batch.commit)
    return {"thread_id": thread_id, "user": target, "messages": messages}


@app.get("/api/messaging/inbox")
async def messaging_inbox(request: Request):
    """Return one inbox row per direct person or human-work job."""
    actor = await _human_actor(request)
    if not db:
        return {"threads": []}

    threads = []
    user_chat_snapshots = await asyncio.to_thread(lambda: list(db.collection("user_chats").stream()))
    user_chat_entries = [(snapshot.id, snapshot.reference) for snapshot in user_chat_snapshots]
    known_thread_ids = {thread_id for thread_id, _ in user_chat_entries}
    # Older releases wrote only the messages subcollection. Include those
    # orphaned threads in the inbox until a normal read/send repairs them.
    try:
        orphan_messages = await asyncio.to_thread(lambda: list(db.collection_group("messages").stream()))
    except Exception:
        orphan_messages = []
    for message_snapshot in orphan_messages:
        path_parts = message_snapshot.reference.path.split("/")
        if len(path_parts) == 4 and path_parts[0] == "user_chats" and path_parts[2] == "messages":
            thread_id = path_parts[1]
            if thread_id not in known_thread_ids:
                user_chat_entries.append((thread_id, db.collection("user_chats").document(thread_id)))
                known_thread_ids.add(thread_id)
    for thread_id, thread_ref in user_chat_entries:
        message_snapshots = await asyncio.to_thread(
            lambda ref=thread_ref.collection("messages"): list(ref.order_by("createdAt").stream())
        )
        if not message_snapshots:
            continue
        other_uid = ""
        messages = []
        unread_count = 0
        for message_snapshot in message_snapshots:
            data = message_snapshot.to_dict() or {}
            sender_uid = str(data.get("sender_uid") or "")
            recipient_uid = str(data.get("recipient_uid") or "")
            if sender_uid != actor["uid"]:
                other_uid = sender_uid
            elif recipient_uid != actor["uid"]:
                other_uid = recipient_uid
            if recipient_uid == actor["uid"] and not data.get("readAt"):
                unread_count += 1
            data["id"] = message_snapshot.id
            messages.append(_human_public(data))
        if not other_uid:
            continue
        try:
            contact = await _user_chat_target(other_uid)
        except HTTPException:
            contact = {"uid": other_uid, "name": "Contact", "email": "", "role": "client"}
        latest = messages[-1]
        latest_sender = "You" if latest.get("sender_uid") == actor["uid"] else contact.get("name") or contact.get("email") or "Contact"
        threads.append({
            "id": f"user:{other_uid}",
            "kind": "user",
            "user": contact,
            "title": contact.get("name") or contact.get("email") or "Contact",
            "role": contact.get("role") or "client",
            "job": None,
            "latest": {"id": latest.get("id"), "senderName": latest_sender, "preview": latest.get("message") or "Attachment", "createdAt": latest.get("createdAt")},
            "latestAt": latest.get("createdAt"),
            "unreadCount": unread_count,
        })

    job_snapshots = await asyncio.to_thread(lambda: list(db.collection(HUMAN_JOB_COLLECTION).stream()))
    for job_snapshot in job_snapshots:
        job = job_snapshot.to_dict() or {}
        job_id = job_snapshot.id
        has_access = actor["role"] == "admin" or job.get("client_uid") == actor["uid"] or job.get("worker_uid") == actor["uid"]
        if not has_access:
            continue
        message_snapshots = await asyncio.to_thread(
            lambda ref=job_snapshot.reference.collection("messages"): list(ref.order_by("createdAt").stream())
        )
        if not message_snapshots:
            continue
        messages = []
        unread_count = 0
        for message_snapshot in message_snapshots:
            data = message_snapshot.to_dict() or {}
            read_by = data.get("readBy") or []
            if data.get("sender_uid") != actor["uid"] and actor["uid"] not in read_by:
                unread_count += 1
            data["id"] = message_snapshot.id
            messages.append(_human_public(data))
        latest = messages[-1]
        source = str(job.get("source_type") or "human_transcription").replace("_", " ").title()
        job_title = job.get("title") or job.get("name") or f"{source} · {job_id[:8]}"
        participant_uid = job.get("client_uid") if actor["role"] == "admin" else (job.get("worker_uid") or "")
        participant = None
        if participant_uid:
            try:
                participant = await _user_chat_target(participant_uid)
            except HTTPException:
                participant = None
        latest_role = str(latest.get("sender_role") or "client").lower()
        latest_sender = "TypeMyworDz admin" if latest_role == "admin" else (latest.get("sender_email") or latest_role.title())
        threads.append({
            "id": f"job:{job_id}",
            "kind": "job",
            "user": participant or {"uid": participant_uid, "name": latest_sender, "email": latest.get("sender_email") or "", "role": latest_role},
            "title": participant.get("name") if participant else latest_sender,
            "role": participant.get("role") if participant else latest_role,
            "job": {"id": job_id, "title": job_title, "status": job.get("status") or "pending"},
            "latest": {"id": latest.get("id"), "senderName": latest_sender, "preview": latest.get("message") or "Attachment", "createdAt": latest.get("createdAt")},
            "latestAt": latest.get("createdAt"),
            "unreadCount": unread_count,
        })

    threads.sort(key=lambda item: str(item.get("latestAt") or ""), reverse=True)
    return {"threads": threads}


@app.get("/api/messaging/unread-count")
async def messaging_unread_count(request: Request):
    """Count unread direct and job-specific messages for the signed-in account."""
    actor = await _human_actor(request)
    if not db:
        return {"count": 0, "direct_count": 0, "job_count": 0}
    direct_unread = 0
    job_unread = 0
    thread_snapshots = await asyncio.to_thread(lambda: list(db.collection("user_chats").stream()))
    for thread_snapshot in thread_snapshots:
        message_ref = thread_snapshot.reference.collection("messages")
        messages = await asyncio.to_thread(lambda ref=message_ref: list(ref.stream()))
        direct_unread += sum(
            1 for message in messages
            if (message.to_dict() or {}).get("recipient_uid") == actor["uid"]
            and not (message.to_dict() or {}).get("readAt")
        )

    job_snapshots = await asyncio.to_thread(lambda: list(db.collection(HUMAN_JOB_COLLECTION).stream()))
    for job_snapshot in job_snapshots:
        job = job_snapshot.to_dict() or {}
        has_access = actor["role"] == "admin" or job.get("client_uid") == actor["uid"] or job.get("worker_uid") == actor["uid"]
        if not has_access:
            continue
        message_ref = job_snapshot.reference.collection("messages")
        messages = await asyncio.to_thread(lambda ref=message_ref: list(ref.stream()))
        job_unread += sum(
            1 for message in messages
            if (message.to_dict() or {}).get("sender_uid") != actor["uid"]
            and actor["uid"] not in ((message.to_dict() or {}).get("readBy") or [])
        )
    return {"count": direct_unread + job_unread, "direct_count": direct_unread, "job_count": job_unread}


@app.post("/api/user-chats/{other_uid}/messages")
async def user_chat_send(other_uid: str, request: Request, attachment: UploadFile = File(None), message: str = Form("")):
    actor = await _user_chat_actor(request)
    target = await _user_chat_target(other_uid)
    if target["uid"] == actor["uid"]:
        raise HTTPException(status_code=400, detail="You cannot message yourself.")
    text = (message or "").strip()
    thread_id = _user_chat_thread_id(actor["uid"], target["uid"])
    attachment_meta = await _user_chat_file(thread_id, attachment) if attachment else None
    if not text and not attachment_meta:
        raise HTTPException(status_code=400, detail="Write a message or attach a file.")
    # Materialize the parent thread before writing its first message. Without
    # this document Firestore only shows a phantom path created by the
    # subcollection, and inbox queries over user_chats cannot discover it.
    thread_ref = db.collection("user_chats").document(thread_id)
    await asyncio.to_thread(thread_ref.set, {
        "participants": [actor["uid"], target["uid"]],
        "participant_emails": [actor["email"], target["email"]],
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "lastMessageAt": firestore.SERVER_TIMESTAMP,
    }, merge=True)
    item = {
        "sender_uid": actor["uid"],
        "sender_email": actor["email"],
        "recipient_uid": target["uid"],
        "message": text[:12000],
        "attachment": attachment_meta,
        "createdAt": firestore.SERVER_TIMESTAMP,
    }
    msg_ref = thread_ref.collection("messages").document()
    await asyncio.to_thread(msg_ref.set, item)
    # Firestore resolves SERVER_TIMESTAMP only after the write. Read the saved
    # document back before returning it so the browser receives a real,
    # JSON-serializable timestamp instead of the sentinel object.
    saved_snapshot = await asyncio.to_thread(msg_ref.get)
    saved_item = saved_snapshot.to_dict() or item
    saved_item["id"] = msg_ref.id
    return {"message": _human_public(saved_item), "user": target, "thread_id": thread_id}


@app.get("/api/user-chats/{other_uid}/messages/{message_id}/attachment")
async def user_chat_attachment(other_uid: str, message_id: str, request: Request):
    actor = await _user_chat_actor(request)
    target = await _user_chat_target(other_uid)
    thread_id = _user_chat_thread_id(actor["uid"], target["uid"])
    snapshot = await asyncio.to_thread(db.collection("user_chats").document(thread_id).collection("messages").document(message_id).get)
    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="That attachment was not found.")
    data = snapshot.to_dict() or {}
    meta = data.get("attachment") or {}
    path = meta.get("storage_path")
    if not path or not str(path).startswith(f"user-chats/{thread_id}/"):
        raise HTTPException(status_code=403, detail="That attachment does not belong to this conversation.")
    bucket = _human_bucket()
    blob = bucket.blob(path) if bucket else None
    if blob is None or not blob.exists():
        raise HTTPException(status_code=404, detail="That attachment is no longer available.")
    raw = await asyncio.to_thread(blob.download_as_bytes)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(meta.get("name") or "attachment")) or "attachment"
    return Response(content=raw, media_type=meta.get("content_type") or "application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{filename}"'})

if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly...")
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting enhanced transcription service on {host}:{port}")
    logger.info("🚀 NEW ENHANCED FEATURES:")
    logger.info(f"  ✅ Smart service selection with updated three-tier logic")
    logger.info(f"  ✅ Three-tier automatic fallback system")
    logger.info(f"  ✅ Admin email-based service prioritization")
    logger.info(f"  ✅ Dedicated AssemblyAI tester logic")
    logger.info(f"  ✅ Dedicated Deepgram tester logic ({DEEPGRAM_TESTER_EMAIL})")
    logger.info(f"  ✅ Speaker diarization for AssemblyAI and Deepgram")
    logger.info(f"  ✅ Dynamic TypeMyworDz1 model selection (nano for free, best for paid)")
    logger.info("  ✅ Unified transcription processing pipeline")
    logger.info("  ✅ Enhanced error handling and service resilience")
    logger.info("  ✅ Paystack payment integration")
    logger.info("  ✅ Multi-language support")
    logger.info("  ✅ Formatted Word document generation")
    logger.info(f"  ✅ User-driven AI features (summarization, Q&A, and bullet points) via TypeMyworDz AI (Anthropic)")
    logger.info(f"  ✅ Admin-driven AI formatting via TypeMyworDz AI (Anthropic) and Google Gemini")
    logger.info(f"  ✅ Google Gemini integration for AI queries - NOW AVAILABLE FOR ALL PAID AI USERS")
    logger.info(f"  ✅ AI Assistant features restricted to paid users (Three-Day, One-Week, Monthly Plan, Yearly Plan plans)")
    logger.info("  🆕 UPDATED: Google Gemini now accessible to ALL paid AI users, not just admins")
    
    logger.info("🔧 NEW TRANSCRIPTION LOGIC:")
    logger.info(f"  - Free trial: Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}")
    logger.info(f"  - Three-Day Plan: Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}")
    logger.info(f"  - One-Week Plan: Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}")
    logger.info(f"  - Monthly Plan: Primary={TYPEMYWORDZ2_NAME} → Fallback1={TYPEMYWORDZ1_NAME} → Fallback2={DEEPGRAM_NAME}")
    logger.info(f"  - Yearly Plan & Admins ({', '.join(ADMIN_EMAILS)}): Primary={TYPEMYWORDZ1_NAME} → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}")
    logger.info(f"  - Speaker Labels requested: Always use {TYPEMYWORDZ1_NAME} first → Fallback1={TYPEMYWORDZ2_NAME} → Fallback2={DEEPGRAM_NAME}")
    logger.info(f"  - Dedicated OpenAI Tester ({OPENAI_TESTER_EMAIL}): Primary={TYPEMYWORDZ2_NAME} (no fallback)")
    logger.info(f"  - Free users: {TYPEMYWORDZ1_NAME} nano model")
    logger.info(f"  - Paid users: {TYPEMYWORDZ1_NAME} best model")
    logger.info(f"  - {TYPEMYWORDZ1_NAME}: AssemblyAI")
    logger.info(f"  - {TYPEMYWORDZ2_NAME}: OpenAI Whisper-1 (typically does NOT support speaker labels)")
    logger.info(f"  - {DEEPGRAM_NAME}: Deepgram")
    logger.info(f"  - {TYPEMYWORDZ_AI_NAME} (Anthropic Claude 3 Haiku / 3.5 Haiku) for user AI text processing")
    logger.info(f"  - Google Gemini for AI text processing - NOW AVAILABLE FOR ALL PAID AI USERS")
    
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
    logger.info(f"Ready to handle requests with {TYPEMYWORDZ1_NAME} + {TYPEMYWORDZ2_NAME} + {DEEPGRAM_NAME} + {TYPEMYWORDZ_AI_NAME} (Anthropic) + Google Gemini integration")
