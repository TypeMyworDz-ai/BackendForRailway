"""One-off copy of Firestore data, Firebase Auth accounts and stored files into Postgres.

Admin-only. It reads from Google (the original Firebase project) and writes into
the Railway database, so it must run while the Google project still works.
Running it again is safe: rows are upserted, and mode="replace" first empties
the target tables so the copy is an exact mirror.
"""
import datetime as _dt
import json
import logging
import threading
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List

import pgcompat as pg

logger = logging.getLogger(__name__)
router = APIRouter()

_state = {"running": False, "log": [], "started": None, "finished": None, "counts": {}, "error": None}
_ctx = {}


def _log(message):
    logger.info("pg-migrate: %s", message)
    _state["log"].append("%s %s" % (_dt.datetime.utcnow().strftime("%H:%M:%S"), message))
    _state["log"] = _state["log"][-200:]


def _plain(value):
    """Turn Firestore-specific types into values pgcompat can store."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (_dt.datetime, _dt.date, bytes, str, int, float, bool)) or value is None:
        return value
    path = getattr(value, "path", None)
    if isinstance(path, str):
        return path
    lat, lng = getattr(value, "latitude", None), getattr(value, "longitude", None)
    if lat is not None and lng is not None:
        return {"latitude": lat, "longitude": lng}
    return str(value)


def _upsert_docs(rows):
    if not rows:
        return
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            pg.psycopg2.extras.execute_values(
                cur,
                "INSERT INTO fs_docs (path, coll, cname, id, data, created_at, updated_at) VALUES %s "
                "ON CONFLICT (path) DO UPDATE SET data=EXCLUDED.data, updated_at=EXCLUDED.updated_at",
                rows,
                template="(%s,%s,%s,%s,%s::jsonb,%s,%s)",
                page_size=100,
            )


def _copy_collection(col_ref, counts):
    batch = []
    for snap in col_ref.stream():
        path = snap.reference.path
        parts = path.split("/")
        data = _plain(snap.to_dict() or {})
        created = snap.create_time or _dt.datetime.now(_dt.timezone.utc)
        updated = snap.update_time or created
        batch.append((path, "/".join(parts[:-1]), parts[-2], parts[-1], json.dumps(pg._encode(data)), created, updated))
        key = parts[0] if len(parts) == 2 else "/".join(parts[:-1])
        counts[key] = counts.get(key, 0) + 1
        if len(batch) >= 100:
            _upsert_docs(batch)
            batch = []
        try:
            for sub in snap.reference.collections():
                _copy_collection(sub, counts)
        except Exception as exc:
            _log("could not list subcollections of %s: %s" % (path, exc))
    _upsert_docs(batch)


def _copy_docs():
    db = _ctx["fs_db"]
    counts = _state["counts"].setdefault("docs", {})
    for col in db.collections():
        _log("copying collection %s" % col.id)
        _copy_collection(col, counts)
        _log("collection %s done (%s docs)" % (col.id, counts.get(col.id, 0)))


def _copy_auth():
    auth = _ctx["fb_auth"]
    total = 0
    page = auth.list_users()
    while page:
        rows = []
        for user in page.users:
            providers = [p.provider_id for p in (user.provider_data or [])]
            meta = user.user_metadata
            created = _dt.datetime.fromtimestamp(meta.creation_timestamp / 1000, _dt.timezone.utc) if meta and meta.creation_timestamp else None
            last = _dt.datetime.fromtimestamp(meta.last_sign_in_timestamp / 1000, _dt.timezone.utc) if meta and meta.last_sign_in_timestamp else None
            rows.append((
                user.uid, (user.email or "").strip().lower() or None, bool(user.email_verified), user.display_name,
                user.photo_url, getattr(user, "password_hash", None), getattr(user, "password_salt", None),
                bool(user.disabled), json.dumps(providers), created, last,
            ))
        with pg._Conn() as conn:
            with conn.cursor() as cur:
                for r in rows:
                    cur.execute(
                        """INSERT INTO auth_users (uid,email,email_verified,display_name,photo_url,fb_hash,fb_salt,disabled,providers,created_at,last_login_at)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,COALESCE(%s,now()),%s)
                           ON CONFLICT (uid) DO UPDATE SET email=EXCLUDED.email, email_verified=EXCLUDED.email_verified,
                             display_name=EXCLUDED.display_name, photo_url=EXCLUDED.photo_url,
                             fb_hash=CASE WHEN auth_users.password_hash IS NULL THEN EXCLUDED.fb_hash ELSE auth_users.fb_hash END,
                             fb_salt=CASE WHEN auth_users.password_hash IS NULL THEN EXCLUDED.fb_salt ELSE auth_users.fb_salt END,
                             disabled=EXCLUDED.disabled, providers=EXCLUDED.providers,
                             last_login_at=GREATEST(auth_users.last_login_at, EXCLUDED.last_login_at)""",
                        r,
                    )
        total += len(rows)
        _log("auth users copied: %d" % total)
        page = page.get_next_page()
    _state["counts"]["auth_users"] = total


def _copy_blobs():
    total = size = 0
    for bucket_fn, prefix in _ctx["fb_buckets"]:
        bucket = bucket_fn()
        if bucket is None:
            continue
        for blob in bucket.list_blobs():
            if blob.name.endswith("/"):
                continue
            data = blob.download_as_bytes()
            pg.Blob(blob.name, prefix).upload_from_string(data, content_type=getattr(blob, "content_type", None))
            total += 1
            size += len(data)
            if total % 20 == 0:
                _log("files copied: %d (%.1f MB)" % (total, size / 1048576.0))
    _state["counts"]["blobs"] = {"files": total, "megabytes": round(size / 1048576.0, 1)}


def _run(mode, parts):
    _state.update({"running": True, "log": [], "started": time.time(), "finished": None, "counts": {}, "error": None})
    try:
        pg.init_schema()
        if mode == "replace":
            with pg._Conn() as conn:
                with conn.cursor() as cur:
                    if "docs" in parts:
                        cur.execute("TRUNCATE fs_docs")
                    if "blobs" in parts:
                        cur.execute("TRUNCATE fs_blobs")
            _log("target tables emptied for: %s" % ",".join(parts))
        if "docs" in parts:
            _copy_docs()
        if "auth" in parts:
            _copy_auth()
        if "blobs" in parts:
            _copy_blobs()
        _log("finished")
    except Exception as exc:
        logger.exception("pg-migrate failed")
        _state["error"] = str(exc)
        _log("FAILED: %s" % exc)
    finally:
        _state["running"] = False
        _state["finished"] = time.time()


class StartBody(BaseModel):
    mode: str = "merge"
    parts: List[str] = ["docs", "auth", "blobs"]


@router.post("/api/admin/pg-migrate/start")
async def start(body: StartBody, request: Request):
    _ctx["require_admin"](request)
    if _state["running"]:
        return {"started": False, "reason": "already running"}
    parts = [p for p in body.parts if p in ("docs", "auth", "blobs")]
    threading.Thread(target=_run, args=(body.mode, parts), daemon=True).start()
    return {"started": True, "parts": parts, "mode": body.mode}


@router.get("/api/admin/pg-migrate/status")
async def status(request: Request):
    _ctx["require_admin"](request)
    return _state


@router.get("/api/admin/pg-migrate/summary")
async def summary(request: Request):
    """Row counts in Postgres (no Google reads)."""
    _ctx["require_admin"](request)
    pg.init_schema()
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT coll, count(*) FROM fs_docs GROUP BY coll ORDER BY coll")
            colls = {r[0]: r[1] for r in cur.fetchall()}
            cur.execute("SELECT count(*) FROM auth_users")
            users = cur.fetchone()[0]
            cur.execute("SELECT count(*), COALESCE(sum(length(data)),0) FROM fs_blobs")
            blobs = cur.fetchone()
            cur.execute("SELECT pg_database_size(current_database())")
            size = cur.fetchone()[0]
    return {"collections": colls, "auth_users": users, "blobs": {"files": blobs[0], "bytes": int(blobs[1])}, "db_bytes": size}


def register(app, fs_db, fb_auth, fb_buckets, require_admin):
    _ctx.update({"fs_db": fs_db, "fb_auth": fb_auth, "fb_buckets": fb_buckets, "require_admin": require_admin})
    app.include_router(router)
