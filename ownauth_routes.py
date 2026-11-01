"""Sign-in routes that do not depend on Google or Firebase.

Accounts live in the Railway Postgres table auth_users (see pgcompat.py).
The error codes mirror the ones Firebase used so the existing sign-in screens
keep showing the right messages.
"""
import hashlib
import json
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import pgcompat as pg

router = APIRouter()

APP_URL = os.environ.get("APP_URL", "https://typemywordz.ai").rstrip("/")
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$")

_DISPOSABLE = None
_send_email = None  # set by register()
_attempts = {}  # key -> [timestamps]


def _disposable_domains():
    global _DISPOSABLE
    if _DISPOSABLE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "disposable_domains.txt")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                _DISPOSABLE = {line.strip().lower() for line in handle if line.strip() and not line.startswith("#")}
        except Exception:
            _DISPOSABLE = set()
    return _DISPOSABLE


def is_disposable(email):
    domain = email.rsplit("@", 1)[-1].lower()
    blocked = _disposable_domains()
    parts = domain.split(".")
    return any(".".join(parts[i:]) in blocked for i in range(len(parts) - 1))


def _fail(code, message, status=400):
    raise HTTPException(status_code=status, detail={"code": code, "message": message})


def _throttle(key, limit=8, window=900):
    now = time.time()
    hits = [t for t in _attempts.get(key, []) if now - t < window]
    if len(hits) >= limit:
        _fail("auth/too-many-requests", "Too many attempts. Please wait a few minutes and try again.", 429)
    hits.append(now)
    _attempts[key] = hits
    if len(_attempts) > 5000:
        for k in list(_attempts)[:2500]:
            _attempts.pop(k, None)


def _public_user(row):
    return {
        "uid": row["uid"],
        "email": row["email"],
        "emailVerified": bool(row["email_verified"]),
        "displayName": row["display_name"] or "",
        "photoURL": row["photo_url"] or "",
    }


def _find(where, params):
    with pg._Conn() as conn:
        with conn.cursor(cursor_factory=pg.psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM auth_users WHERE " + where, params)
            return cur.fetchone()


def _session(row):
    token = pg.make_token(row["uid"], row["email"], row["email_verified"], row["display_name"], row["token_version"])
    return {"token": token, "user": _public_user(row)}


def _client_ip(request):
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")) or "?"


class RegisterBody(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginBody(BaseModel):
    email: str
    password: str


class EmailBody(BaseModel):
    email: str


class ResetBody(BaseModel):
    token: str
    password: str


class TokenBody(BaseModel):
    token: str


class ChangePasswordBody(BaseModel):
    current_password: str = ""
    new_password: str


class ProfileBody(BaseModel):
    display_name: str = ""


def _issue_link_token(uid, purpose, hours):
    raw = secrets.token_urlsafe(32)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO auth_tokens (token_hash, uid, purpose, expires_at) VALUES (%s,%s,%s,%s)",
                (digest, uid, purpose, datetime.now(timezone.utc) + timedelta(hours=hours)),
            )
    return raw


def _consume_link_token(raw, purpose):
    digest = hashlib.sha256((raw or "").encode()).hexdigest()
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE auth_tokens SET used_at=now() WHERE token_hash=%s AND purpose=%s AND used_at IS NULL "
                "AND expires_at > now() RETURNING uid",
                (digest, purpose),
            )
            row = cur.fetchone()
    if not row:
        _fail("auth/invalid-action-code", "This link has expired or was already used. Please request a new one.")
    return row[0]


def _check_password(password):
    if len(password or "") < 8:
        _fail("auth/weak-password", "Password should be at least 8 characters.")
    if len(password) > 200:
        _fail("auth/weak-password", "Password is too long.")


async def _mail(address, subject, html, text, label):
    if _send_email is None:
        return {"sent": False}
    return await _send_email(address, subject, html, text, label)


def _email_shell(title, body_html, button_text, link):
    html = (
        '<div style="font-family:Arial,Helvetica,sans-serif;max-width:520px;margin:0 auto;padding:24px;color:#1f2937">'
        '<h2 style="margin:0 0 12px;color:#5b21b6">%s</h2>%s'
        '<p style="margin:24px 0"><a href="%s" style="background:#16a34a;color:#fff;text-decoration:none;'
        'padding:12px 22px;border-radius:8px;font-weight:600">%s</a></p>'
        '<p style="font-size:13px;color:#6b7280">If the button does not work, copy this link into your browser:<br>%s</p>'
        '<p style="font-size:13px;color:#6b7280">If you did not ask for this, you can ignore this email.</p></div>'
    ) % (title, body_html, link, button_text, link)
    return html


@router.post("/auth/register")
async def register_account(body: RegisterBody, request: Request):
    email = (body.email or "").strip().lower()
    if not EMAIL_RE.match(email) or len(email) > 254:
        _fail("auth/invalid-email", "That email address does not look right.")
    _check_password(body.password)
    _throttle("reg:" + _client_ip(request), limit=10, window=3600)
    if is_disposable(email):
        _fail("auth/disposable-email", "Temporary email addresses are not accepted. Please use your regular email.")
    if _find("lower(email)=%s", (email,)):
        _fail("auth/email-already-in-use", "An account with this email already exists. Try signing in instead.")
    name = (body.name or "").strip()[:120]
    pg.create_user(email=email, password=body.password, display_name=name or None)
    row = _find("lower(email)=%s", (email,))
    return _session(row)


@router.post("/auth/login")
async def login_account(body: LoginBody, request: Request):
    email = (body.email or "").strip().lower()
    _throttle("login:%s:%s" % (_client_ip(request), email), limit=10, window=900)
    row = _find("lower(email)=%s", (email,))
    generic = ("auth/invalid-credential", "The email or password is not correct.")
    if not row:
        _fail(*generic, status=401)
    ok, rehash = pg.verify_password(body.password or "", row)
    if not ok:
        if not row.get("password_hash") and not row.get("fb_hash"):
            _fail("auth/no-password", "This account has no password yet. Use 'Forgot password' to create one.", 401)
        _fail(*generic, status=401)
    if row["disabled"]:
        _fail("auth/user-disabled", "This account has been disabled. Contact support.", 403)
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            if rehash:
                cur.execute("UPDATE auth_users SET password_hash=%s, fb_hash=NULL, fb_salt=NULL, last_login_at=now() WHERE uid=%s",
                            (pg.hash_password(body.password), row["uid"]))
            else:
                cur.execute("UPDATE auth_users SET last_login_at=now() WHERE uid=%s", (row["uid"],))
    row = _find("uid=%s", (row["uid"],))
    return _session(row)


def _current(request):
    header = request.headers.get("authorization", "")
    token = header.split(" ", 1)[1].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        _fail("auth/no-token", "Sign-in is required.", 401)
    try:
        claims = pg.verify_id_token(token)
    except Exception:
        _fail("auth/token-expired", "Your sign-in session has expired.", 401)
    row = _find("uid=%s", (claims["uid"],))
    if not row:
        _fail("auth/user-not-found", "Account not found.", 401)
    return row


@router.get("/auth/me")
async def me(request: Request):
    return _session(_current(request))


@router.post("/auth/refresh")
async def refresh(request: Request):
    return _session(_current(request))


@router.post("/auth/forgot")
async def forgot_password(body: EmailBody, request: Request):
    email = (body.email or "").strip().lower()
    _throttle("forgot:" + _client_ip(request), limit=6, window=3600)
    row = _find("lower(email)=%s", (email,)) if EMAIL_RE.match(email) else None
    if row and not row["disabled"]:
        raw = _issue_link_token(row["uid"], "reset", 1)
        link = "%s/?reset=%s" % (APP_URL, raw)
        html = _email_shell("Reset your password", "<p>Use the button below to choose a new password. The link works for one hour.</p>",
                            "Choose a new password", link)
        text = "Choose a new password using this link (valid for one hour):\n%s\n" % link
        await _mail(row["email"], "Reset your TypeMyworDz password", html, text, "Password reset email")
    return {"ok": True}


@router.post("/auth/reset")
async def reset_password(body: ResetBody):
    _check_password(body.password)
    uid = _consume_link_token(body.token, "reset")
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE auth_users SET password_hash=%s, fb_hash=NULL, fb_salt=NULL, token_version=token_version+1, "
                "email_verified=true WHERE uid=%s",
                (pg.hash_password(body.password), uid),
            )
    return _session(_find("uid=%s", (uid,)))


@router.post("/auth/send-verification")
async def send_verification(request: Request):
    row = _current(request)
    if row["email_verified"]:
        return {"ok": True, "alreadyVerified": True}
    raw = _issue_link_token(row["uid"], "verify", 72)
    link = "%s/?verify=%s" % (APP_URL, raw)
    html = _email_shell("Confirm your email", "<p>Confirm this address so we can reach you about your transcripts.</p>",
                        "Confirm my email", link)
    text = "Confirm your email using this link:\n%s\n" % link
    await _mail(row["email"], "Confirm your TypeMyworDz email", html, text, "Verification email")
    return {"ok": True}


@router.post("/auth/verify")
async def verify_email(body: TokenBody):
    uid = _consume_link_token(body.token, "verify")
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE auth_users SET email_verified=true WHERE uid=%s", (uid,))
    return _session(_find("uid=%s", (uid,)))


@router.post("/auth/change-password")
async def change_password(body: ChangePasswordBody, request: Request):
    row = _current(request)
    _check_password(body.new_password)
    if row.get("password_hash") or row.get("fb_hash"):
        ok, _ = pg.verify_password(body.current_password or "", row)
        if not ok:
            _fail("auth/wrong-password", "Your current password is not correct.", 401)
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE auth_users SET password_hash=%s, fb_hash=NULL, fb_salt=NULL, token_version=token_version+1 WHERE uid=%s",
                        (pg.hash_password(body.new_password), row["uid"]))
    return _session(_find("uid=%s", (row["uid"],)))


@router.post("/auth/update-profile")
async def update_profile(body: ProfileBody, request: Request):
    row = _current(request)
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE auth_users SET display_name=%s WHERE uid=%s", ((body.display_name or "").strip()[:120] or None, row["uid"]))
    return _session(_find("uid=%s", (row["uid"],)))


class GoogleBody(BaseModel):
    credential: str


def _google_get(url, headers=None):
    import urllib.request, json as _json
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return _json.loads(resp.read().decode("utf-8"))


@router.post("/auth/google")
async def google_sign_in(body: GoogleBody, request: Request):
    """Sign in with a Google access token. Google only vouches for the email address;
    the session itself is ours, so every other part of the app is Google-free."""
    import os, urllib.parse
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        _fail("auth/operation-not-allowed", "Google sign-in is not switched on.", 400)
    _throttle("google:" + _client_ip(request), limit=30, window=900)
    try:
        info = _google_get("https://oauth2.googleapis.com/tokeninfo?access_token=" + urllib.parse.quote(body.credential))
        if info.get("aud") != client_id and info.get("azp") != client_id:
            _fail("auth/invalid-credential", "That Google sign-in was not meant for this app.", 401)
        profile = _google_get("https://www.googleapis.com/oauth2/v3/userinfo", {"Authorization": "Bearer " + body.credential})
    except HTTPException:
        raise
    except Exception:
        _fail("auth/network-request-failed", "Could not confirm your Google sign-in. Please try again.", 502)
    email = (profile.get("email") or "").strip().lower()
    if not email or str(profile.get("email_verified")).lower() != "true":
        _fail("auth/invalid-credential", "Your Google email address is not verified.", 401)
    row = _find("lower(email)=%s", (email,))
    if row and row["disabled"]:
        _fail("auth/user-disabled", "This account has been disabled. Contact support.", 403)
    if not row:
        pg.create_user(email=email, password=None, display_name=(profile.get("name") or "")[:120] or None,
                       email_verified=True, providers=["google.com"])
        row = _find("lower(email)=%s", (email,))
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE auth_users SET email_verified=true, last_login_at=now(), "
                "photo_url=COALESCE(photo_url,%s), display_name=COALESCE(NULLIF(display_name,''),%s), "
                "providers=CASE WHEN providers @> '\"google.com\"'::jsonb THEN providers ELSE providers || '\"google.com\"'::jsonb END "
                "WHERE uid=%s",
                (profile.get("picture"), profile.get("name"), row["uid"]),
            )
    return _session(_find("uid=%s", (row["uid"],)))


def register(app, send_email):
    global _send_email
    _send_email = send_email
    app.include_router(router)
