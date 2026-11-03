"""Postgres-backed stand-ins for the parts of Firebase the backend uses.

The rest of the application was written against the Firestore, Firebase Auth and
Firebase Storage Python APIs. This module keeps those call shapes but stores
everything in the Railway Postgres database, so nothing depends on Google
billing. It is only used when DATABASE_URL is set and USE_POSTGRES is not "0".
"""
import base64
import copy
import datetime as _dt
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid

import psycopg2
import psycopg2.extras
import psycopg2.pool

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# --------------------------------------------------------------------------
# Connection pool and schema
# --------------------------------------------------------------------------
_pool = None
_pool_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS fs_docs (
    path       text PRIMARY KEY,
    coll       text NOT NULL,
    cname      text NOT NULL,
    id         text NOT NULL,
    data       jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS fs_docs_coll_idx ON fs_docs (coll);
CREATE INDEX IF NOT EXISTS fs_docs_cname_idx ON fs_docs (cname);
CREATE INDEX IF NOT EXISTS fs_docs_data_idx ON fs_docs USING gin (data jsonb_path_ops);

CREATE TABLE IF NOT EXISTS fs_blobs (
    path         text PRIMARY KEY,
    data         bytea NOT NULL,
    content_type text,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS auth_users (
    uid            text PRIMARY KEY,
    email          text,
    email_verified boolean NOT NULL DEFAULT false,
    display_name   text,
    photo_url      text,
    password_hash  text,
    fb_hash        text,
    fb_salt        text,
    disabled       boolean NOT NULL DEFAULT false,
    providers      jsonb NOT NULL DEFAULT '[]'::jsonb,
    token_version  integer NOT NULL DEFAULT 0,
    created_at     timestamptz NOT NULL DEFAULT now(),
    last_login_at  timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS auth_users_email_idx ON auth_users (lower(email));

CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash text PRIMARY KEY,
    uid        text NOT NULL,
    purpose    text NOT NULL,
    expires_at timestamptz NOT NULL,
    used_at    timestamptz
);
"""


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                if not DATABASE_URL:
                    raise RuntimeError("DATABASE_URL is not set")
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    1, int(os.environ.get("PG_POOL_MAX", "40")), DATABASE_URL, connect_timeout=15
                )
                conn = _pool.getconn()
                try:
                    with conn.cursor() as cur:
                        cur.execute(SCHEMA)
                    conn.commit()
                finally:
                    _pool.putconn(conn)
    return _pool


class _Conn:
    """Context manager: borrow a connection, commit on success, roll back on error."""

    def __init__(self, conn=None):
        self._external = conn is not None
        self.conn = conn

    def __enter__(self):
        if not self._external:
            deadline = time.time() + 20
            while True:
                try:
                    self.conn = _get_pool().getconn()
                    break
                except psycopg2.pool.PoolError:
                    # Every connection is busy: wait for one instead of failing the request.
                    if time.time() > deadline:
                        raise
                    time.sleep(0.05)
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if self._external:
            return False
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            try:
                _get_pool().putconn(self.conn)
            except Exception:
                pass
        return False


# --------------------------------------------------------------------------
# Sentinels, exceptions, value encoding
# --------------------------------------------------------------------------
class _Sentinel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


SERVER_TIMESTAMP = _Sentinel("SERVER_TIMESTAMP")
DELETE_FIELD = _Sentinel("DELETE_FIELD")


class Increment:
    def __init__(self, value):
        self.value = value


class ArrayUnion:
    def __init__(self, values):
        self.values = list(values)


class ArrayRemove:
    def __init__(self, values):
        self.values = list(values)


class NotFound(Exception):
    pass


class AlreadyExists(Exception):
    pass


class Query:
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"


class FieldFilter:
    def __init__(self, field_path, op_string, value):
        self.field_path = field_path
        self.op_string = op_string
        self.value = value


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc)


def _norm_dt(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=_dt.timezone.utc)
    return value.astimezone(_dt.timezone.utc)


def _encode(value):
    """Python value -> JSON-safe structure (sentinels must already be resolved)."""
    if isinstance(value, _dt.datetime):
        return {"__dt": _norm_dt(value).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")}
    if isinstance(value, _dt.date):
        return {"__d": value.isoformat()}
    if isinstance(value, (bytes, bytearray)):
        return {"__b": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_encode(v) for v in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    return value


def _decode(value):
    if isinstance(value, dict):
        if len(value) == 1:
            if "__dt" in value:
                return _dt.datetime.strptime(value["__dt"], "%Y-%m-%dT%H:%M:%S.%f+00:00").replace(tzinfo=_dt.timezone.utc)
            if "__d" in value:
                return _dt.date.fromisoformat(value["__d"])
            if "__b" in value:
                return base64.b64decode(value["__b"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _get_path(data, field_path):
    node = data
    for part in str(field_path).split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return (False, None)
    return (True, node)


def _set_path(data, field_path, value):
    parts = str(field_path).split(".")
    node = data
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _del_path(data, field_path):
    parts = str(field_path).split(".")
    node = data
    for part in parts[:-1]:
        node = node.get(part)
        if not isinstance(node, dict):
            return
    node.pop(parts[-1], None)


def _resolve(current, field_path, value, now):
    """Return the stored (decoded-python) value for a write, or DELETE_FIELD."""
    if value is DELETE_FIELD:
        return DELETE_FIELD
    if value is SERVER_TIMESTAMP:
        return now
    if isinstance(value, Increment):
        ok, existing = _get_path(current, field_path)
        base = existing if ok and isinstance(existing, (int, float)) and not isinstance(existing, bool) else 0
        return base + value.value
    if isinstance(value, ArrayUnion):
        ok, existing = _get_path(current, field_path)
        arr = list(existing) if ok and isinstance(existing, list) else []
        for item in value.values:
            if item not in arr:
                arr.append(item)
        return arr
    if isinstance(value, ArrayRemove):
        ok, existing = _get_path(current, field_path)
        arr = list(existing) if ok and isinstance(existing, list) else []
        return [item for item in arr if item not in value.values]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            r = _resolve(current, "%s.%s" % (field_path, k) if field_path else k, v, now)
            if r is not DELETE_FIELD:
                out[k] = r
        return out
    return value


def _apply_set(current, data, merge, now):
    if not merge:
        out = {}
        for k, v in data.items():
            r = _resolve({}, k, v, now)
            if r is not DELETE_FIELD:
                out[k] = r
        return out
    out = copy.deepcopy(current)

    def merge_into(target, src, prefix):
        for k, v in src.items():
            path = "%s.%s" % (prefix, k) if prefix else k
            if isinstance(v, dict):
                sub = target.get(k)
                if not isinstance(sub, dict):
                    sub = {}
                    target[k] = sub
                merge_into(sub, v, path)
            else:
                r = _resolve(out, path, v, now)
                if r is DELETE_FIELD:
                    target.pop(k, None)
                else:
                    target[k] = r

    merge_into(out, data, "")
    return out


def _apply_update(current, updates, now):
    out = copy.deepcopy(current)
    for field_path, value in updates.items():
        r = _resolve(out, field_path, value, now)
        if r is DELETE_FIELD:
            _del_path(out, field_path)
        else:
            _set_path(out, field_path, r)
    return out


def _new_id():
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(secrets.choice(alphabet) for _ in range(20))


# --------------------------------------------------------------------------
# Low-level document store
# --------------------------------------------------------------------------
def _split(path):
    parts = [p for p in str(path).strip("/").split("/") if p]
    return parts


def _lock(cur, path):
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (path,))


def _row_to_snapshot(db, row):
    path, data, created_at, updated_at = row
    return DocumentSnapshot(db, path, _decode(data), True, created_at, updated_at)


def _fetch_doc(cur, path):
    cur.execute("SELECT data, created_at, updated_at FROM fs_docs WHERE path=%s", (path,))
    return cur.fetchone()


def _write_doc(cur, path, data_py, existing):
    parts = _split(path)
    coll = "/".join(parts[:-1])
    cname = parts[-2]
    doc_id = parts[-1]
    payload = json.dumps(_encode(data_py))
    cur.execute(
        """INSERT INTO fs_docs (path, coll, cname, id, data) VALUES (%s,%s,%s,%s,%s::jsonb)
           ON CONFLICT (path) DO UPDATE SET data=EXCLUDED.data, updated_at=now()""",
        (path, coll, cname, doc_id, payload),
    )


def _op_set(cur, path, data, merge):
    _lock(cur, path)
    row = _fetch_doc(cur, path)
    current = _decode(row[0]) if row else {}
    _write_doc(cur, path, _apply_set(current, data, merge, _utcnow()), row)


def _op_update(cur, path, updates):
    _lock(cur, path)
    row = _fetch_doc(cur, path)
    if not row:
        raise NotFound("No document to update: %s" % path)
    _write_doc(cur, path, _apply_update(_decode(row[0]), updates, _utcnow()), row)


def _op_create(cur, path, data):
    _lock(cur, path)
    if _fetch_doc(cur, path):
        raise AlreadyExists("Document already exists: %s" % path)
    _write_doc(cur, path, _apply_set({}, data, False, _utcnow()), None)


def _op_delete(cur, path):
    cur.execute("DELETE FROM fs_docs WHERE path=%s", (path,))


# --------------------------------------------------------------------------
# Public Firestore-like classes
# --------------------------------------------------------------------------
class DocumentSnapshot:
    def __init__(self, db, path, data, exists, created_at=None, updated_at=None):
        self._db = db
        self._path = path
        self._data = data if exists else None
        self.exists = exists
        self.create_time = created_at
        self.update_time = updated_at
        self.read_time = _utcnow()

    @property
    def id(self):
        return _split(self._path)[-1]

    @property
    def reference(self):
        return DocumentReference(self._db, self._path)

    def to_dict(self):
        return copy.deepcopy(self._data) if self.exists else None

    def get(self, field_path):
        if not self.exists:
            return None
        ok, value = _get_path(self._data, field_path)
        return copy.deepcopy(value) if ok else None

    def __getitem__(self, field_path):
        ok, value = _get_path(self._data or {}, field_path)
        if not ok:
            raise KeyError(field_path)
        return copy.deepcopy(value)


class DocumentReference:
    def __init__(self, db, path):
        self._db = db
        self.path = "/".join(_split(path))

    @property
    def id(self):
        return _split(self.path)[-1]

    @property
    def parent(self):
        return CollectionReference(self._db, "/".join(_split(self.path)[:-1]))

    def collection(self, name):
        return CollectionReference(self._db, self.path + "/" + name.strip("/"))

    def get(self, transaction=None, **_):
        if transaction is not None:
            return transaction.get(self)
        with _Conn() as conn:
            with conn.cursor() as cur:
                row = _fetch_doc(cur, self.path)
        if not row:
            return DocumentSnapshot(self._db, self.path, None, False)
        return DocumentSnapshot(self._db, self.path, _decode(row[0]), True, row[1], row[2])

    def set(self, data, merge=False, **_):
        with _Conn() as conn:
            with conn.cursor() as cur:
                _op_set(cur, self.path, data, merge)

    def update(self, updates, **_):
        with _Conn() as conn:
            with conn.cursor() as cur:
                _op_update(cur, self.path, updates)

    def create(self, data, **_):
        with _Conn() as conn:
            with conn.cursor() as cur:
                _op_create(cur, self.path, data)

    def delete(self, **_):
        with _Conn() as conn:
            with conn.cursor() as cur:
                _op_delete(cur, self.path)

    def __eq__(self, other):
        return isinstance(other, DocumentReference) and other.path == self.path

    def __hash__(self):
        return hash(self.path)


_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _comparable(a, b):
    if isinstance(a, _dt.datetime) and isinstance(b, _dt.datetime):
        return _norm_dt(a), _norm_dt(b)
    return a, b


def _match(data, doc_id, flt):
    field, op, value = flt
    if field == "__name__":
        found, actual = True, doc_id
    else:
        found, actual = _get_path(data, field)
    if op == "array_contains":
        return found and isinstance(actual, list) and value in actual
    if op == "array_contains_any":
        return found and isinstance(actual, list) and any(v in actual for v in value)
    if op == "in":
        return found and actual in list(value)
    if op == "not-in":
        return found and actual not in list(value)
    if not found:
        return False
    try:
        a, b = _comparable(actual, value)
        if op in ("<", "<=", ">", ">=") and (a is None or b is None):
            return False
        if op in ("<", "<=", ">", ">=") and type(a) is not type(b):
            num = (int, float)
            if not (isinstance(a, num) and isinstance(b, num)):
                return False
        return _OPS[op](a, b)
    except TypeError:
        return False


def _sort_key(value):
    # Firestore order: null < bool < number < timestamp < string < bytes < array < map
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, float)):
        return (2, value)
    if isinstance(value, _dt.datetime):
        return (3, _norm_dt(value).timestamp())
    if isinstance(value, str):
        return (4, value)
    return (5, json.dumps(_encode(value), sort_keys=True, default=str))


class Query_:
    pass


class _QueryBase:
    def __init__(self, db, coll, group=False, filters=None, orders=None, limit=None, offset=0, select=None):
        self._db = db
        self._coll = coll
        self._group = group
        self._filters = filters or []
        self._orders = orders or []
        self._limit = limit
        self._offset = offset
        self._select = select

    def _clone(self, **kw):
        args = dict(db=self._db, coll=self._coll, group=self._group, filters=list(self._filters),
                    orders=list(self._orders), limit=self._limit, offset=self._offset, select=self._select)
        args.update(kw)
        return _QueryBase(**args)

    def where(self, field_path=None, op_string=None, value=None, filter=None):
        if filter is not None:
            field_path, op_string, value = filter.field_path, filter.op_string, filter.value
        return self._clone(filters=self._filters + [(field_path, op_string, value)])

    def order_by(self, field_path, direction="ASCENDING"):
        return self._clone(orders=self._orders + [(field_path, direction)])

    def limit(self, count):
        return self._clone(limit=count)

    def offset(self, count):
        return self._clone(offset=count)

    def select(self, field_paths):
        return self._clone(select=list(field_paths))

    def _run(self, conn=None):
        sql = "SELECT path, data, created_at, updated_at FROM fs_docs WHERE "
        params = []
        if self._group:
            sql += "cname=%s"
            params.append(self._coll)
        else:
            sql += "coll=%s"
            params.append(self._coll)
        contain = {}
        for field, op, value in self._filters:
            if op == "==" and field != "__name__" and isinstance(value, (str, bool, int, float)) and not isinstance(value, _Sentinel):
                parts = field.split(".")
                node = contain
                for p in parts[:-1]:
                    node = node.setdefault(p, {})
                node[parts[-1]] = value
        if contain:
            sql += " AND data @> %s::jsonb"
            params.append(json.dumps(contain))
        with _Conn(conn) as c:
            with c.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        results = []
        for row in rows:
            data = _decode(row[1])
            doc_id = _split(row[0])[-1]
            if all(_match(data, doc_id, f) for f in self._filters):
                if all(_get_path(data, o[0])[0] for o in self._orders):
                    results.append((row, data))
        for field, direction in reversed(self._orders):
            results.sort(key=lambda item: _sort_key(_get_path(item[1], field)[1]), reverse=(direction == "DESCENDING"))
        if not self._orders:
            results.sort(key=lambda item: _split(item[0][0])[-1])
        if self._offset:
            results = results[self._offset:]
        if self._limit is not None:
            results = results[: self._limit]
        snaps = []
        for row, data in results:
            if self._select is not None:
                projected = {}
                for fld in self._select:
                    ok, val = _get_path(data, fld)
                    if ok:
                        _set_path(projected, fld, val)
                data = projected
            snaps.append(DocumentSnapshot(self._db, row[0], data, True, row[2], row[3]))
        return snaps

    def stream(self, transaction=None, **_):
        if transaction is not None:
            return iter(transaction.get(self))
        return iter(self._run())

    def get(self, transaction=None, **_):
        if transaction is not None:
            return transaction.get(self)
        return self._run()

    def count(self, alias=None):
        outer = self

        class _Agg:
            def get(self_inner, **_):
                n = len(outer._run())
                return [[type("AggregationResult", (), {"value": n, "alias": alias})()]]

        return _Agg()


class CollectionReference(_QueryBase):
    def __init__(self, db, path):
        path = "/".join(_split(path))
        super().__init__(db, path)
        self.path = path

    @property
    def id(self):
        return _split(self.path)[-1]

    @property
    def parent(self):
        parts = _split(self.path)
        return DocumentReference(self._db, "/".join(parts[:-1])) if len(parts) > 1 else None

    def document(self, document_id=None):
        return DocumentReference(self._db, self.path + "/" + (document_id or _new_id()))

    def add(self, data):
        ref = self.document()
        ref.create(data)
        return (_utcnow(), ref)

    def list_documents(self):
        return [s.reference for s in self._run()]


class Transaction:
    def __init__(self, db):
        self._db = db
        self._conn = None
        self._writes = []

    def _begin(self):
        self._conn = _Conn().__enter__()
        self._conn.autocommit = False
        self._writes = []

    def _end(self, commit):
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            if commit:
                conn.commit()
            else:
                conn.rollback()
        finally:
            _get_pool().putconn(conn)

    def get(self, target):
        if self._conn is None:
            self._begin()
        if isinstance(target, DocumentReference):
            with self._conn.cursor() as cur:
                _lock(cur, target.path)
                row = _fetch_doc(cur, target.path)
            if not row:
                return DocumentSnapshot(self._db, target.path, None, False)
            return DocumentSnapshot(self._db, target.path, _decode(row[0]), True, row[1], row[2])
        return target._run(conn=self._conn)

    def set(self, ref, data, merge=False):
        self._writes.append(("set", ref.path, data, merge))

    def update(self, ref, updates, **_):
        self._writes.append(("update", ref.path, updates))

    def create(self, ref, data):
        self._writes.append(("create", ref.path, data))

    def delete(self, ref, **_):
        self._writes.append(("delete", ref.path))

    def _flush(self):
        if self._conn is None:
            self._begin()
        with self._conn.cursor() as cur:
            for op in self._writes:
                if op[0] == "set":
                    _op_set(cur, op[1], op[2], op[3])
                elif op[0] == "update":
                    _op_update(cur, op[1], op[2])
                elif op[0] == "create":
                    _op_create(cur, op[1], op[2])
                elif op[0] == "delete":
                    _op_delete(cur, op[1])
        self._writes = []


def transactional(fn):
    def wrapper(transaction, *args, **kwargs):
        last = None
        for attempt in range(5):
            transaction._begin()
            try:
                result = fn(transaction, *args, **kwargs)
                transaction._flush()
                transaction._end(True)
                return result
            except (psycopg2.extensions.TransactionRollbackError, psycopg2.OperationalError) as exc:
                transaction._end(False)
                last = exc
                time.sleep(0.05 * (attempt + 1))
            except BaseException:
                transaction._end(False)
                raise
        raise last

    return wrapper


class WriteBatch:
    def __init__(self, db):
        self._db = db
        self._writes = []

    def set(self, ref, data, merge=False):
        self._writes.append(("set", ref.path, data, merge))

    def update(self, ref, updates, **_):
        self._writes.append(("update", ref.path, updates))

    def create(self, ref, data):
        self._writes.append(("create", ref.path, data))

    def delete(self, ref, **_):
        self._writes.append(("delete", ref.path))

    def commit(self):
        with _Conn() as conn:
            with conn.cursor() as cur:
                for op in self._writes:
                    if op[0] == "set":
                        _op_set(cur, op[1], op[2], op[3])
                    elif op[0] == "update":
                        _op_update(cur, op[1], op[2])
                    elif op[0] == "create":
                        _op_create(cur, op[1], op[2])
                    elif op[0] == "delete":
                        _op_delete(cur, op[1])
        self._writes = []
        return []


class Client:
    def collection(self, path):
        return CollectionReference(self, path)

    def document(self, path):
        return DocumentReference(self, path)

    def collection_group(self, name):
        return _QueryBase(self, name, group=True)

    def transaction(self, **_):
        return Transaction(self)

    def batch(self):
        return WriteBatch(self)

    def get_all(self, refs, **_):
        return [r.get() for r in refs]

    def collections(self):
        with _Conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT coll FROM fs_docs WHERE position('/' in coll)=0")
                return [CollectionReference(self, r[0]) for r in cur.fetchall()]


def client():
    return Client()


class _FirestoreNamespace:
    SERVER_TIMESTAMP = SERVER_TIMESTAMP
    DELETE_FIELD = DELETE_FIELD
    Increment = Increment
    ArrayUnion = ArrayUnion
    ArrayRemove = ArrayRemove
    Query = Query
    transactional = staticmethod(transactional)
    client = staticmethod(client)


firestore = _FirestoreNamespace()


# --------------------------------------------------------------------------
# Auth: password hashing, tokens, user records
# --------------------------------------------------------------------------
class UserNotFoundError(Exception):
    pass


class _UserMetadata:
    def __init__(self, created, last):
        self.creation_timestamp = int(created.timestamp() * 1000) if created else None
        self.last_sign_in_timestamp = int(last.timestamp() * 1000) if last else None


class UserRecord:
    def __init__(self, row):
        (self.uid, self.email, self.email_verified, self.display_name, self.photo_url,
         self.disabled, providers, created, last) = row
        self.provider_data = [type("P", (), {"provider_id": p})() for p in (providers or [])]
        self.user_metadata = _UserMetadata(created, last)
        self.custom_claims = None


_USER_COLS = "uid,email,email_verified,display_name,photo_url,disabled,providers,created_at,last_login_at"

JWT_SECRET = os.environ.get("AUTH_JWT_SECRET", "")
TOKEN_TTL_SECONDS = int(os.environ.get("AUTH_TOKEN_TTL_SECONDS", str(60 * 60 * 24 * 30)))


def _b64u(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_dec(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password):
    salt = secrets.token_bytes(16)
    n, r, p = 2 ** 14, 8, 1
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32, maxmem=64 * 1024 * 1024)
    return "scrypt$%d$%d$%d$%s$%s" % (n, r, p, _b64u(salt), _b64u(dk))


def _b64_any(value):
    """Decode standard or URL-safe base64, with or without padding."""
    value = (value or "").strip().replace("-", "+").replace("_", "/")
    value += "=" * (-len(value) % 4)
    return base64.b64decode(value)


def _firebase_scrypt_check(password, fb_hash, fb_salt):
    """Verify a password against a hash exported from Firebase Auth.

    Firebase uses scrypt with parameters shown in the console, then AES-256-CTR
    encrypts the signer key with the derived key. Parameters come from env vars.
    """
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    signer_key = _b64_any(os.environ["FIREBASE_HASH_SIGNER_KEY"])
    salt_sep = _b64_any(os.environ["FIREBASE_HASH_SALT_SEPARATOR"])
    rounds = int(os.environ.get("FIREBASE_HASH_ROUNDS", "8"))
    mem_cost = int(os.environ.get("FIREBASE_HASH_MEM_COST", "14"))
    salt = _b64_any(fb_salt) + salt_sep
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2 ** mem_cost, r=rounds, p=1,
                        dklen=32, maxmem=256 * 1024 * 1024)
    cipher = Cipher(algorithms.AES(dk), modes.CTR(b"\x00" * 16))
    enc = cipher.encryptor()
    candidate = enc.update(signer_key) + enc.finalize()
    return hmac.compare_digest(candidate, _b64_any(fb_hash))


def verify_password(password, row):
    """row: dict with password_hash, fb_hash, fb_salt. Returns (ok, needs_rehash)."""
    stored = row.get("password_hash") or ""
    if stored.startswith("scrypt$"):
        try:
            _, n, r, p, salt, dk = stored.split("$")
            calc = hashlib.scrypt(password.encode("utf-8"), salt=_b64u_dec(salt), n=int(n), r=int(r), p=int(p),
                                  dklen=32, maxmem=64 * 1024 * 1024)
            return hmac.compare_digest(calc, _b64u_dec(dk)), False
        except Exception:
            return False, False
    if row.get("fb_hash") and row.get("fb_salt"):
        try:
            return _firebase_scrypt_check(password, row["fb_hash"], row["fb_salt"]), True
        except Exception:
            logger.exception("Firebase hash check failed")
            return False, False
    return False, False


def make_token(uid, email, email_verified=False, name="", token_version=0, ttl=None):
    if not JWT_SECRET:
        raise RuntimeError("AUTH_JWT_SECRET is not set")
    now = int(time.time())
    payload = {
        "uid": uid, "user_id": uid, "sub": uid, "email": email or "",
        "email_verified": bool(email_verified), "name": name or "",
        "tv": int(token_version), "iat": now, "exp": now + int(ttl or TOKEN_TTL_SECONDS),
        "iss": "typemywordz",
    }
    head = _b64u(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64u(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(JWT_SECRET.encode(), ("%s.%s" % (head, body)).encode(), hashlib.sha256).digest()
    return "%s.%s.%s" % (head, body, _b64u(sig))


def verify_id_token(token, **_):
    try:
        head, body, sig = token.split(".")
        expected = hmac.new(JWT_SECRET.encode(), ("%s.%s" % (head, body)).encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64u_dec(sig)):
            raise ValueError("bad signature")
        payload = json.loads(_b64u_dec(body))
    except Exception:
        raise ValueError("Invalid token")
    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("Token expired")
    with _Conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT disabled, token_version FROM auth_users WHERE uid=%s", (payload["uid"],))
            row = cur.fetchone()
    if not row or row[0] or int(row[1]) != int(payload.get("tv", 0)):
        raise ValueError("Account unavailable")
    return payload


def _user_from(where_sql, params):
    with _Conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT %s FROM auth_users WHERE %s" % (_USER_COLS, where_sql), params)
            row = cur.fetchone()
    if not row:
        raise UserNotFoundError("No user record")
    return UserRecord(row)


def get_user(uid, **_):
    return _user_from("uid=%s", (uid,))


def get_user_by_email(email, **_):
    return _user_from("lower(email)=lower(%s)", (email,))


def delete_user(uid, **_):
    with _Conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM auth_users WHERE uid=%s", (uid,))
            if cur.rowcount == 0:
                raise UserNotFoundError("No user record")


def create_user(uid=None, email=None, password=None, display_name=None, email_verified=False, providers=None):
    uid = uid or uuid.uuid4().hex[:28]
    pw = hash_password(password) if password else None
    with _Conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO auth_users (uid,email,email_verified,display_name,password_hash,providers) "
                "VALUES (%s,%s,%s,%s,%s,%s::jsonb)",
                (uid, email, email_verified, display_name, pw, json.dumps(providers or ["password"])),
            )
    return get_user(uid)


class _AuthNamespace:
    UserNotFoundError = UserNotFoundError
    verify_id_token = staticmethod(verify_id_token)
    get_user = staticmethod(get_user)
    get_user_by_email = staticmethod(get_user_by_email)
    delete_user = staticmethod(delete_user)
    create_user = staticmethod(create_user)


firebase_auth = _AuthNamespace()


# --------------------------------------------------------------------------
# Storage (replaces the Firebase Storage bucket)
# --------------------------------------------------------------------------
class Blob:
    def __init__(self, path, prefix=""):
        self.name = path
        self._key = prefix + path

    def upload_from_string(self, data, content_type=None, **_):
        if isinstance(data, str):
            data = data.encode("utf-8")
        with _Conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO fs_blobs (path, data, content_type) VALUES (%s,%s,%s) "
                    "ON CONFLICT (path) DO UPDATE SET data=EXCLUDED.data, content_type=EXCLUDED.content_type, created_at=now()",
                    (self._key, psycopg2.Binary(bytes(data)), content_type),
                )

    def upload_from_filename(self, filename, content_type=None, **_):
        with open(filename, "rb") as handle:
            self.upload_from_string(handle.read(), content_type=content_type)

    def download_as_bytes(self, **_):
        with _Conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT data FROM fs_blobs WHERE path=%s", (self._key,))
                row = cur.fetchone()
        if not row:
            raise NotFound("Blob not found: %s" % self.name)
        return bytes(row[0])

    def exists(self, **_):
        with _Conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM fs_blobs WHERE path=%s", (self._key,))
                return cur.fetchone() is not None

    def delete(self, **_):
        with _Conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM fs_blobs WHERE path=%s", (self._key,))


class Bucket:
    def __init__(self, prefix=""):
        self.name = "postgres"
        self._prefix = prefix

    def blob(self, path, **_):
        return Blob(path, self._prefix)

    def list_blobs(self, prefix="", **_):
        full = self._prefix + prefix
        like = full.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with _Conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT path FROM fs_blobs WHERE path LIKE %s ORDER BY path", (like,))
                return [Blob(r[0][len(self._prefix):], self._prefix) for r in cur.fetchall()]


class _StorageNamespace:
    @staticmethod
    def bucket(name=None):
        # The training-asset bucket was a separate Google bucket; keep its files apart.
        return Bucket("training/" if name and "train" in str(name).lower() else "")


firebase_storage = _StorageNamespace()


def init_schema():
    _get_pool()
