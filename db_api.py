"""Small, rule-guarded document API used by the web app.

The browser used to talk to Firestore directly. These routes give it the same
operations through the backend, with explicit access rules:

  users/{uid}            the owner and admins
  transcriptions/{id}    the owner (userId) and admins
  askChats/{id}          the owner (userId) and admins
  feedback/{id}          any signed-in user may add; only admins read or edit
  admin_stats/*          admins only
  anything else          admins only
"""
import datetime as _dt

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Any, List, Optional

import pgcompat as pg

router = APIRouter()
_admin_emails = lambda: set()


def _deny(status=403, message="You do not have access to that."):
    raise HTTPException(status_code=status, detail=message)


def _caller(request):
    header = request.headers.get("authorization", "")
    token = header.split(" ", 1)[1].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        _deny(401, "Sign-in is required.")
    try:
        claims = pg.verify_id_token(token)
    except Exception:
        _deny(401, "Your sign-in session has expired.")
    email = (claims.get("email") or "").strip().lower()
    claims["_admin"] = email in {e.lower() for e in _admin_emails()}
    return claims


def _from_client(value):
    """Client JSON -> python values, resolving the small set of tagged types."""
    if isinstance(value, dict):
        if len(value) == 1:
            if "__dt" in value:
                text = str(value["__dt"]).replace("Z", "+00:00")
                return pg._norm_dt(_dt.datetime.fromisoformat(text))
            if "__sv" in value:
                return pg.SERVER_TIMESTAMP
            if "__del" in value:
                return pg.DELETE_FIELD
            if "__inc" in value:
                return pg.Increment(value["__inc"])
            if "__au" in value:
                return pg.ArrayUnion([_from_client(v) for v in value["__au"]])
        return {k: _from_client(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_client(v) for v in value]
    return value


def _to_client(value):
    return pg._encode(value)


def _parts(path):
    parts = [p for p in str(path or "").strip("/").split("/") if p]
    if len(parts) < 2 or len(parts) % 2 != 0:
        _deny(400, "Bad document path.")
    return parts


def _existing(path):
    snap = pg.client().document(path).get()
    return snap.to_dict() if snap.exists else None


def _authorize(caller, op, path, data=None):
    """Raise 403 unless this caller may perform op on path."""
    if caller["_admin"]:
        return
    parts = _parts(path)
    if len(parts) != 2:
        _deny()
    coll, doc_id = parts
    uid = caller["uid"]
    if coll == "users":
        if doc_id != uid:
            _deny()
        return
    if coll in ("transcriptions", "askChats"):
        current = _existing(path)
        if op == "read":
            if current is not None and current.get("userId") != uid:
                _deny()
            return
        if current is not None and current.get("userId") != uid:
            _deny()
        if op in ("set", "create") and (data or {}).get("userId") not in ((None, uid) if current is not None else (uid,)):
            _deny()
        return
    if coll == "feedback":
        if op in ("create",) or (op == "set" and _existing(path) is None):
            if (data or {}).get("userId") not in (None, uid):
                _deny()
            return
        _deny()
    _deny()


class GetBody(BaseModel):
    path: str


class QueryBody(BaseModel):
    collection: str
    filters: List[List[Any]] = []
    orders: List[List[str]] = []
    limit: Optional[int] = None


class Write(BaseModel):
    op: str  # set | update | delete | add
    path: str
    data: Optional[dict] = None
    merge: bool = False


class Check(BaseModel):
    path: str
    exists: bool


class CommitBody(BaseModel):
    writes: List[Write]
    checks: List[Check] = []


def _snap_json(snap):
    return {"id": snap.id, "path": snap.reference.path, "exists": bool(snap.exists),
            "data": _to_client(snap.to_dict()) if snap.exists else None}


@router.post("/db/get")
async def db_get(body: GetBody, request: Request):
    caller = _caller(request)
    _parts(body.path)
    _authorize(caller, "read", body.path)
    return _snap_json(pg.client().document(body.path).get())


@router.post("/db/query")
async def db_query(body: QueryBody, request: Request):
    caller = _caller(request)
    coll = body.collection.strip("/")
    q = pg.client().collection(coll)
    filters = [(f[0], f[1], _from_client(f[2])) for f in body.filters if len(f) == 3]
    if not caller["_admin"]:
        if coll not in ("transcriptions", "askChats"):
            _deny()
        if not any(f[0] == "userId" and f[1] == "==" and f[2] == caller["uid"] for f in filters):
            filters.append(("userId", "==", caller["uid"]))
    for field, op, value in filters:
        q = q.where(field, op, value)
    for order in body.orders:
        q = q.order_by(order[0], direction=(order[1] if len(order) > 1 else "ASCENDING"))
    if body.limit:
        q = q.limit(min(int(body.limit), 5000))
    snaps = q.get()
    if not caller["_admin"]:
        snaps = [s for s in snaps if (s.to_dict() or {}).get("userId") == caller["uid"]]
    return {"docs": [_snap_json(s) for s in snaps]}


@router.post("/db/commit")
async def db_commit(body: CommitBody, request: Request):
    caller = _caller(request)
    db = pg.client()
    plan = []
    created_ids = []
    for w in body.writes:
        data = _from_client(w.data) if w.data is not None else None
        path = w.path
        if w.op == "add":
            ref = db.collection(path.strip("/")).document()
            path = ref.path
            created_ids.append(ref.id)
            _authorize(caller, "create", path, data)
            plan.append(("create", path, data, False))
            continue
        _parts(path)
        if w.op not in ("set", "update", "delete"):
            _deny(400, "Unknown operation.")
        _authorize(caller, w.op, path, data)
        plan.append((w.op, path, data, w.merge))
    with pg._Conn() as conn:
        with conn.cursor() as cur:
            for check in body.checks:
                pg._lock(cur, check.path)
                exists = pg._fetch_doc(cur, check.path) is not None
                if exists != check.exists:
                    raise HTTPException(status_code=409, detail="conflict")
            for op, path, data, merge in plan:
                if op == "set":
                    pg._op_set(cur, path, data or {}, merge)
                elif op == "update":
                    pg._op_update(cur, path, data or {})
                elif op == "create":
                    pg._op_create(cur, path, data or {})
                elif op == "delete":
                    pg._op_delete(cur, path)
                else:
                    raise HTTPException(status_code=400, detail="Unknown operation.")
    return {"ok": True, "ids": created_ids}


def register(app, admin_emails):
    global _admin_emails
    _admin_emails = admin_emails
    app.include_router(router)
