"""Production-data-free tests for persistent notification ownership and state."""
import ast
import asyncio
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class FakeHTTPException(Exception):
    def __init__(self, status_code=500, detail=""):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeSnapshot:
    def __init__(self, document_id, data, reference, exists=True):
        self.id = document_id
        self._data = dict(data or {})
        self.reference = reference
        self.exists = exists

    def to_dict(self):
        return dict(self._data)


class FakeNotificationRef:
    def __init__(self, database, document_id):
        self.database = database
        self.id = document_id

    def get(self):
        data = self.database.documents.get(self.id)
        return FakeSnapshot(self.id, data or {}, self, exists=data is not None)

    def set(self, data, merge=False):
        if merge:
            self.database.documents.setdefault(self.id, {}).update(dict(data))
        else:
            self.database.documents[self.id] = dict(data)


class FakeBatch:
    def __init__(self, database):
        self.database = database
        self.writes = []

    def set(self, reference, data, merge=False):
        self.writes.append((reference, dict(data), merge))

    def commit(self):
        for reference, data, merge in self.writes:
            reference.set(data, merge=merge)
        return []


class FakeNotificationQuery:
    def __init__(self, database, expected_uid):
        self.database = database
        self.expected_uid = expected_uid

    def stream(self):
        return [
            FakeSnapshot(document_id, data, FakeNotificationRef(self.database, document_id))
            for document_id, data in self.database.documents.items()
            if data.get("recipient_uid") == self.expected_uid
        ]


class FakeNotificationCollection:
    def __init__(self, database):
        self.database = database

    def document(self, document_id):
        return FakeNotificationRef(self.database, document_id)

    def where(self, *, filter):
        field, operator, expected = filter
        if (field, operator) != ("recipient_uid", "=="):
            raise AssertionError(f"Unexpected notification query: {filter}")
        return FakeNotificationQuery(self.database, expected)


class FakeNotificationDB:
    def __init__(self):
        self.documents = {}

    def collection(self, name):
        if name != "user_notifications":
            raise AssertionError(f"Unexpected collection: {name}")
        return FakeNotificationCollection(self)

    def batch(self):
        return FakeBatch(self)


class FakeFirestore:
    SERVER_TIMESTAMP = "SERVER_TIMESTAMP"


async def _fake_actor(_request):
    return {"uid": _NAMESPACE["actor_uid"], "email": "test@example.test", "role": "worker"}


_NAMESPACE = {"actor_uid": "user-a"}


def _load_functions():
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    wanted = {
        "_user_notification_ref",
        "_create_user_notification",
        "_update_user_notification_states",
        "mark_application_notification_read",
        "snooze_application_notification",
        "record_application_notification_ring",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            node.decorator_list = []
            nodes.append(node)
    namespace = {
        "asyncio": asyncio,
        "uuid": uuid,
        "datetime": datetime,
        "timedelta": timedelta,
        "db": FakeNotificationDB(),
        "firestore": FakeFirestore(),
        "FieldFilter": lambda field, operator, value: (field, operator, value),
        "USER_NOTIFICATION_COLLECTION": "user_notifications",
        "HTTPException": FakeHTTPException,
        "Request": object,
        "_human_actor": _fake_actor,
        "_NAMESPACE": _NAMESPACE,
        "logger": type("Logger", (), {"warning": staticmethod(lambda *args, **kwargs: None)})(),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
    return namespace


class NotificationStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = _load_functions()

    def setUp(self):
        self.functions["db"] = FakeNotificationDB()
        _NAMESPACE["actor_uid"] = "user-a"

    def test_retry_preserves_created_read_and_snooze_state(self):
        create = self.functions["_create_user_notification"]
        database = self.functions["db"]
        asyncio.run(create("user-a", "event-1", "assignment", "New assignment", "Open Work Room.", route="human_worker", job_id="job-1", target_id="part-1", created_at="first-created"))
        record = next(iter(database.documents.values()))
        record["readAt"] = "already-read"
        record["snoozedUntil"] = "snoozed"
        asyncio.run(create("user-a", "event-1", "assignment", "New assignment", "Updated safe text.", route="human_worker", job_id="job-1", target_id="part-1", created_at="retry-created"))
        self.assertEqual(len(database.documents), 1)
        saved = next(iter(database.documents.values()))
        self.assertEqual(saved["createdAt"], "first-created")
        self.assertEqual(saved["readAt"], "already-read")
        self.assertEqual(saved["snoozedUntil"], "snoozed")
        self.assertNotIn("message", saved)
        self.assertNotIn("transcript", saved)
        self.assertNotIn("preview", saved)

    def test_state_updates_are_scoped_to_owner_job_kind_and_target(self):
        functions = self.functions
        create = functions["_create_user_notification"]
        update = functions["_update_user_notification_states"]
        database = functions["db"]
        for recipient, event, job, target in [
            ("user-a", "part-one", "job-1", "part-1"),
            ("user-a", "part-two", "job-1", "part-2"),
            ("user-a", "other-job", "job-2", "part-1"),
            ("user-b", "other-user", "job-1", "part-1"),
        ]:
            asyncio.run(create(recipient, event, "assignment", "New assignment", "Open Work Room.", route="human_worker", job_id=job, target_id=target))
        asyncio.run(update("user-a", job_id="job-1", target_id="part-1", kinds={"assignment"}, read=True, completed=True))
        records = list(database.documents.values())
        changed = [item for item in records if item.get("readAt")]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["recipient_uid"], "user-a")
        self.assertEqual(changed[0]["target_id"], "part-1")
        self.assertEqual(changed[0]["actionCompletedAt"], "SERVER_TIMESTAMP")

    def test_read_snooze_and_ring_endpoints_require_recipient_ownership(self):
        functions = self.functions
        create = functions["_create_user_notification"]
        database = functions["db"]
        notification_id = asyncio.run(create("user-a", "event-1", "assignment", "New assignment", "Open Work Room.", route="human_worker", job_id="job-1"))
        _NAMESPACE["actor_uid"] = "user-b"
        with self.assertRaises(FakeHTTPException) as denied:
            asyncio.run(functions["mark_application_notification_read"](notification_id, object()))
        self.assertEqual(denied.exception.status_code, 404)
        _NAMESPACE["actor_uid"] = "user-a"
        asyncio.run(functions["mark_application_notification_read"](notification_id, object()))
        asyncio.run(functions["snooze_application_notification"](notification_id, object()))
        asyncio.run(functions["record_application_notification_ring"](notification_id, object()))
        record = database.documents[notification_id]
        self.assertEqual(record["readAt"], "SERVER_TIMESTAMP")
        self.assertIsNotNone(record["snoozedUntil"])
        self.assertEqual(record["lastRungAt"], "SERVER_TIMESTAMP")


if __name__ == "__main__":
    unittest.main()
