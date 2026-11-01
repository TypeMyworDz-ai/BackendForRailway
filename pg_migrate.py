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


PAGE = 400
LEAF_COLLECTIONS = {"trafficEvents", "transcriptions", "user_notifications", "paddleEvents", "payment_intents"}
PROBE_LIMIT = 3000   # very large collections are only probed for sub-collections in their first docs


def _fetch_page(col_ref, last):
    q = col_ref.order_by("__name__").limit(PAGE)
    if last is not None:
        q = q.start_after(last)
    for attempt in range(5):
        try:
            return list(q.stream())
        except Exception as exc:
            _log("retry %d on %s: %s" % (attempt + 1, col_ref.path, exc))
            time.sleep(2 + attempt * 3)
    raise RuntimeError("could not read " + col_ref.path)


def _copy_collection(col_ref, counts):
    seen = 0
    last = None
    while True:
        snaps = _fetch_page(col_ref, last)
        if not snaps:
            break
        batch = []
        for snap in snaps:
            path = snap.reference.path
            parts = path.split("/")
            data = _plain(snap.to_dict() or {})
            created = snap.create_time or _dt.datetime.now(_dt.timezone.utc)
            updated = snap.update_time or created
            batch.append((path, "/".join(parts[:-1]), parts[-2], parts[-1], json.dumps(pg._encode(data)), created, updated))
            key = parts[0] if len(parts) == 2 else "/".join(parts[:-1])
            counts[key] = counts.get(key, 0) + 1
            seen += 1
            if seen <= PROBE_LIMIT and col_ref.id not in LEAF_COLLECTIONS:
                try:
                    for sub in snap.reference.collections():
                        _copy_collection(sub, counts)
                except Exception as exc:
                    _log("could not list subcollections of %s: %s" % (path, exc))
        _upsert_docs(batch)
        last = snaps[-1]
        if len(snaps) < PAGE:
            break


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


def _guard(request):
    """Admin token, or the one-off MIGRATE_SECRET header (remove the variable afterwards)."""
    import hmac, os
    secret = os.environ.get("MIGRATE_SECRET", "")
    sent = request.headers.get("x-migrate-secret", "")
    if secret and sent and hmac.compare_digest(secret, sent):
        return {"email": "migration"}
    return _ctx["require_admin"](request)


class StartBody(BaseModel):
    mode: str = "merge"
    parts: List[str] = ["docs", "auth", "blobs"]


@router.post("/api/admin/pg-migrate/start")
async def start(body: StartBody, request: Request):
    _guard(request)
    import os
    if os.environ.get("USE_POSTGRES") == "1":
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="The one-off copy is switched off now that the app runs on Postgres.")
    if _state["running"]:
        return {"started": False, "reason": "already running"}
    parts = [p for p in body.parts if p in ("docs", "auth", "blobs")]
    threading.Thread(target=_run, args=(body.mode, parts), daemon=True).start()
    return {"started": True, "parts": parts, "mode": body.mode}


@router.get("/api/admin/pg-migrate/status")
async def status(request: Request):
    _guard(request)
    return _state


@router.get("/api/admin/pg-migrate/summary")
async def summary(request: Request):
    """Row counts in Postgres (no Google reads)."""
    _guard(request)
    pg.init_schema()
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT coll, count(*) FROM fs_docs GROUP BY coll ORDER BY coll")
            colls = {r[0]: r[1] for r in cur.fetchall()}
            cur.execute("SELECT count(*), count(fb_hash), count(password_hash) FROM auth_users")
            ur = cur.fetchone()
            users = {"total": ur[0], "with_firebase_hash": ur[1], "with_own_hash": ur[2]}
            cur.execute("SELECT count(*), COALESCE(sum(length(data)),0) FROM fs_blobs")
            blobs = cur.fetchone()
            cur.execute("SELECT pg_database_size(current_database())")
            size = cur.fetchone()[0]
    return {"collections": colls, "auth_users": users, "blobs": {"files": blobs[0], "bytes": int(blobs[1])}, "db_bytes": size}


# ---------------------------------------------------------------- reconcile
_rec = {"running": False, "report": None, "error": None, "log": []}
RECON_SKIP = {"trafficEvents", "transcriptions"}


def _real_google():
    """A direct Google connection, independent of the Postgres switch."""
    import base64, os
    import firebase_admin
    from firebase_admin import credentials, firestore as real_fs, auth as real_auth
    info = json.loads(base64.b64decode(os.environ["FIREBASE_ADMIN_SDK_CONFIG_BASE64"]).decode("utf-8"))
    try:
        app = firebase_admin.get_app("reconcile")
    except ValueError:
        app = firebase_admin.initialize_app(credentials.Certificate(info), name="reconcile")
    return real_fs.client(app), real_auth, app


def _pg_times(paths):
    if not paths:
        return {}
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT path, updated_at FROM fs_docs WHERE path = ANY(%s)", (list(paths),))
            return {r[0]: r[1] for r in cur.fetchall()}


def _recon_collection(col_ref, apply, report):
    last = None
    while True:
        q = col_ref.order_by("__name__").limit(PAGE)
        if last is not None:
            q = q.start_after(last)
        snaps = None
        for attempt in range(5):
            try:
                snaps = list(q.stream())
                break
            except Exception as exc:
                time.sleep(2 + attempt * 3)
        if snaps is None:
            raise RuntimeError("could not read " + col_ref.path)
        if not snaps:
            break
        have = _pg_times([sn.reference.path for sn in snaps])
        fix = []
        for sn in snaps:
            path = sn.reference.path
            parts = path.split("/")
            key = parts[0] if len(parts) == 2 else "/".join(["*" if i % 2 == 0 and i else x for i, x in enumerate(parts[:-1])])
            entry = report["collections"].setdefault(key, {"google": 0, "missing": 0, "newer_in_google": 0, "fixed": 0, "samples": []})
            entry["google"] += 1
            updated = sn.update_time
            status = None
            if path not in have:
                status = "missing"
            elif updated and have[path] and updated > have[path] + _dt.timedelta(seconds=1):
                status = "newer_in_google"
            if status:
                entry[status] += 1
                if len(entry["samples"]) < 5:
                    entry["samples"].append("%s %s" % (status, path))
                if apply:
                    data = _plain(sn.to_dict() or {})
                    created = sn.create_time or _dt.datetime.now(_dt.timezone.utc)
                    fix.append((path, "/".join(parts[:-1]), parts[-2], parts[-1], json.dumps(pg._encode(data)), created, updated or created))
                    entry["fixed"] += 1
            if col_ref.id not in LEAF_COLLECTIONS:
                try:
                    for sub in sn.reference.collections():
                        _recon_collection(sub, apply, report)
                except Exception:
                    pass
        if fix:
            _upsert_docs(fix)
        last = snaps[-1]
        if len(snaps) < PAGE:
            break


def _recon_run(apply):
    _rec.update({"running": True, "report": None, "error": None})
    report = {"applied": apply, "collections": {}, "auth": {}, "started": time.time()}
    _rec["report"] = report
    try:
        gdb, gauth, app = _real_google()
        for col in gdb.collections():
            if col.id in RECON_SKIP:
                continue
            _recon_collection(col, apply, report)
        # accounts created in Google after the copy
        with pg._Conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT uid FROM auth_users")
                known = {r[0] for r in cur.fetchall()}
        missing = []
        page = gauth.list_users(app=app)
        total = 0
        while page:
            for u in page.users:
                total += 1
                if u.uid not in known:
                    missing.append(u)
            page = page.get_next_page()
        report["auth"] = {"google_accounts": total, "missing_in_postgres": [m.email for m in missing]}
        if apply and missing:
            with pg._Conn() as conn:
                with conn.cursor() as cur:
                    for u in missing:
                        providers = [pi.provider_id for pi in (u.provider_data or [])] or ["password"]
                        cur.execute(
                            "INSERT INTO auth_users (uid,email,email_verified,display_name,photo_url,fb_hash,fb_salt,disabled,providers) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT (uid) DO NOTHING",
                            (u.uid, (u.email or "").lower(), bool(u.email_verified), u.display_name, u.photo_url,
                             getattr(u, "password_hash", None), getattr(u, "password_salt", None), bool(u.disabled), json.dumps(providers)),
                        )
        report["finished"] = time.time()
    except Exception as exc:
        logger.exception("reconcile failed")
        _rec["error"] = str(exc)
    finally:
        _rec["running"] = False


class ReconBody(BaseModel):
    apply: bool = False


@router.post("/api/admin/pg-migrate/reconcile")
async def reconcile(body: ReconBody, request: Request):
    _guard(request)
    if _rec["running"]:
        return {"started": False, "reason": "already running"}
    threading.Thread(target=_recon_run, args=(body.apply,), daemon=True).start()
    return {"started": True, "apply": body.apply}


@router.get("/api/admin/pg-migrate/reconcile")
async def reconcile_status(request: Request):
    _guard(request)
    return _rec


# ---------------------------------------------------------------- clean-up (admin)
_CLEAN = {
    "transcriptions": ("coll = 'transcriptions' OR coll ~ '^users/[^/]+/transcriptions$'", "transcriptions"),
    "notifications": ("coll = 'user_notifications'", "notifications"),
    "messages": ("coll = 'user_chats' OR coll ~ '^user_chats/[^/]+/messages$'", "direct messages"),
}


@router.get("/api/admin/cleanup/counts")
async def cleanup_counts(request: Request):
    _ctx["require_admin"](request)
    out = {}
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            for key, (where, _label) in _CLEAN.items():
                cur.execute("SELECT count(*) FROM fs_docs WHERE " + where)
                out[key] = cur.fetchone()[0]
    return out


class CleanBody(BaseModel):
    what: str
    confirm: str = ""


@router.post("/api/admin/cleanup/delete-all")
async def cleanup_delete_all(body: CleanBody, request: Request):
    admin = _ctx["require_admin"](request)
    if body.what not in _CLEAN:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Unknown item")
    if body.confirm != "DELETE":
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Type DELETE to confirm")
    where, label = _CLEAN[body.what]
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM fs_docs WHERE " + where)
            n = cur.rowcount
    logger.warning("ADMIN CLEANUP: %s deleted %d %s", (admin or {}).get("email"), n, label)
    return {"deleted": n, "what": body.what}


def register(app, fs_db, fb_auth, fb_buckets, require_admin):
    _ctx.update({"fs_db": fs_db, "fb_auth": fb_auth, "fb_buckets": fb_buckets, "require_admin": require_admin})
    app.include_router(router)
