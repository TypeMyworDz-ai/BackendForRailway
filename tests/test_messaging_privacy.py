"""Focused, production-data-free tests for direct-message privacy."""
import ast
import asyncio
import unittest
from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


class FakeHTTPException(Exception):
    def __init__(self, status_code=500, detail=""):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeSnapshot:
    def __init__(self, document_id, data, reference):
        self.id = document_id
        self._data = dict(data)
        self.reference = reference
        self.exists = True

    def to_dict(self):
        return dict(self._data)


class FakeMessageCollection:
    def __init__(self, messages):
        self._messages = messages

    def order_by(self, _field):
        return self

    def stream(self):
        ordered = sorted(self._messages, key=lambda item: str(item[1].get("createdAt") or ""))
        return [FakeSnapshot(message_id, data, None) for message_id, data in ordered]


class FakeParentReference:
    def __init__(self, document_id, data):
        self.id = document_id
        self._data = dict(data)

    def collection(self, _name):
        return FakeMessageCollection(self._data.get("messages", []))


class FakeParentQuery:
    def __init__(self, snapshots):
        self._snapshots = snapshots

    def stream(self):
        return list(self._snapshots)


class FakeParentCollection:
    def __init__(self, documents):
        self._documents = documents

    def where(self, *, filter):
        field, operator, expected = filter
        if operator != "array_contains":
            raise AssertionError(f"Unexpected query operator: {operator}")
        return FakeParentQuery([
            FakeSnapshot(document_id, data, FakeParentReference(document_id, data))
            for document_id, data in self._documents.items()
            if expected in (data.get(field) or [])
        ])

    def stream(self):
        return [
            FakeSnapshot(document_id, data, FakeParentReference(document_id, data))
            for document_id, data in self._documents.items()
        ]


class FakeDatabase:
    def __init__(self, user_chats, human_jobs=None):
        self.user_chats = user_chats
        self.human_jobs = human_jobs or {}

    def collection(self, name):
        if name == "user_chats":
            return FakeParentCollection(self.user_chats)
        if name == "human_jobs":
            return FakeParentCollection(self.human_jobs)
        return FakeParentCollection({})


async def _worker_actor(_request):
    return {"uid": "worker-1", "email": "worker@example.test", "role": "worker"}


async def _admin_actor(_request):
    return {"uid": "admin-1", "email": "typemywordz@gmail.com", "role": "admin"}


async def _target(uid):
    contacts = {
        "admin-1": {"uid": "admin-1", "email": "typemywordz@gmail.com", "name": "Support", "role": "admin"},
        "client-1": {"uid": "client-1", "email": "client@example.test", "name": "Client One", "role": "client"},
    }
    if uid not in contacts:
        raise FakeHTTPException(404, "missing")
    return contacts[uid]


def _load_functions():
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    wanted = {
        "_user_chat_thread_id",
        "_user_chat_other_uid",
        "_user_chat_parent_matches",
        "_user_chat_message_matches_pair",
        "_user_chat_assert_target",
        "_human_job_worker_uids",
        "_human_assert_job_conversation_access",
        "_human_thread_for",
        "messaging_inbox",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            node.decorator_list = []
            nodes.append(node)
    namespace = {
        "asyncio": asyncio,
        "FieldFilter": lambda field, operator, value: (field, operator, value),
        "HUMAN_JOB_COLLECTION": "human_jobs",
        "HTTPException": FakeHTTPException,
        "Request": object,
        "is_admin_user": lambda email: str(email or "").lower() == "typemywordz@gmail.com",
        "_human_actor": _worker_actor,
        "_human_public": lambda data: dict(data or {}),
        "_user_chat_target": _target,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MAIN_PATH), "exec"), namespace)
    return namespace


class MessagingPrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = _load_functions()

    def setUp(self):
        self.functions["_human_actor"] = _worker_actor
        self.functions["db"] = FakeDatabase({})

    def test_inbox_returns_only_participant_threads(self):
        functions = self.functions
        actor_uid = "worker-1"
        own_thread = functions["_user_chat_thread_id"](actor_uid, "admin-1")
        unrelated_thread = functions["_user_chat_thread_id"]("client-1", "client-2")
        mixed_thread = functions["_user_chat_thread_id"](actor_uid, "admin-1")
        database = FakeDatabase({
            own_thread: {
                "participants": [actor_uid, "admin-1"],
                "messages": [
                    ("msg-own", {"sender_uid": "admin-1", "recipient_uid": actor_uid, "message": "private-to-worker", "createdAt": "2026-09-25T10:00:00"}),
                ],
            },
            unrelated_thread: {
                "participants": ["client-1", "client-2"],
                "messages": [
                    ("msg-secret", {"sender_uid": "client-1", "recipient_uid": "client-2", "message": "private-client-content", "createdAt": "2026-09-25T10:01:00"}),
                ],
            },
            # Even if bad legacy data puts this account in the parent list,
            # unrelated message parties are rejected before a preview is built.
            mixed_thread + "-bad": {
                "participants": [actor_uid, "admin-1"],
                "messages": [
                    ("msg-mixed", {"sender_uid": "client-1", "recipient_uid": "client-2", "message": "private-mixed-content", "createdAt": "2026-09-25T10:02:00"}),
                ],
            },
        })
        functions["db"] = database
        result = asyncio.run(functions["messaging_inbox"](object()))
        self.assertEqual([item["id"] for item in result["threads"]], ["user:admin-1"])
        rendered = repr(result)
        self.assertIn("private-to-worker", rendered)
        self.assertNotIn("private-client-content", rendered)
        self.assertNotIn("private-mixed-content", rendered)

    def test_worker_inbox_never_includes_the_client_job_thread(self):
        functions = self.functions
        database = FakeDatabase({}, {
            "job-1": {
                "client_uid": "client-1",
                "segments": [{"worker_uid": "worker-1", "status": "in_progress"}],
                "title": "Human transcript",
                "status": "in_progress",
                "messages": [
                    ("client-side", {"thread": "client", "sender_uid": "admin-1", "sender_role": "admin", "message": "private-client-job-message", "createdAt": "2026-09-25T10:00:00"}),
                    ("worker-side", {"thread": "worker", "sender_uid": "admin-1", "sender_role": "admin", "message": "worker-job-update", "createdAt": "2026-09-25T10:01:00"}),
                ],
            },
        })
        functions["db"] = database
        result = asyncio.run(functions["messaging_inbox"](object()))
        job_thread = next(item for item in result["threads"] if item["id"] == "job:job-1:worker")
        self.assertEqual(job_thread["title"], "TypeMyworDz admin")
        self.assertEqual(job_thread["latest"]["preview"], "worker-job-update")
        self.assertNotIn("private-client-job-message", repr(result))

    def test_admin_inbox_exposes_only_worker_thread_for_split_claimants(self):
        functions = self.functions
        functions["_human_actor"] = _admin_actor
        database = FakeDatabase({}, {
            "job-2": {
                "client_uid": "client-1",
                "assigned_worker_uids": ["worker-1", "worker-2"],
                "segments": [
                    {"worker_uid": "worker-1", "worker_name": "Worker One"},
                    {"worker_uid": "worker-2", "worker_name": "Worker Two"},
                ],
                "title": "Confidential job",
                "status": "split_in_progress",
                "messages": [
                    ("client-msg", {"thread": "client", "sender_uid": "client-1", "sender_role": "client", "message": "legacy-client-only message", "createdAt": "2026-09-25T10:00:00"}),
                    ("worker-msg", {"thread": "worker", "sender_uid": "worker-2", "sender_role": "worker", "message": "worker-only message", "createdAt": "2026-09-25T10:01:00"}),
                ],
            },
        })
        functions["db"] = database
        result = asyncio.run(functions["messaging_inbox"](object()))
        job_threads = [item for item in result["threads"] if item["kind"] == "job"]
        self.assertEqual([item["id"] for item in job_threads], ["job:job-2:worker"])
        self.assertEqual(job_threads[0]["title"], "2 workers")
        self.assertEqual(job_threads[0]["latest"]["preview"], "worker-only message")
        self.assertNotIn("legacy-client-only message", repr(result))

    def test_only_admins_and_assigned_workers_can_use_job_conversations(self):
        functions = self.functions
        allowed_job = {"segments": [{"worker_uid": "worker-1"}]}
        functions["_human_assert_job_conversation_access"](allowed_job, {"uid": "admin-1", "role": "admin"})
        functions["_human_assert_job_conversation_access"](allowed_job, {"uid": "worker-1", "role": "worker"})
        for actor in (
            {"uid": "client-1", "role": "client"},
            {"uid": "worker-2", "role": "worker"},
        ):
            with self.subTest(actor=actor), self.assertRaises(FakeHTTPException):
                functions["_human_assert_job_conversation_access"](allowed_job, actor)

    def test_legacy_client_thread_is_not_selectable_by_any_user(self):
        choose = self.functions["_human_thread_for"]
        self.assertEqual(choose({"role": "admin"}, ""), "worker")
        self.assertEqual(choose({"role": "worker"}, "client"), "worker")
        with self.assertRaises(FakeHTTPException):
            choose({"role": "admin"}, "client")
        with self.assertRaises(FakeHTTPException):
            choose({"role": "client"}, "worker")

    def test_message_must_belong_to_exact_actor_pair_and_path(self):
        functions = self.functions
        pair_id = functions["_user_chat_thread_id"]("worker-1", "admin-1")
        own_message = {"sender_uid": "admin-1", "recipient_uid": "worker-1"}
        other_people_message = {"sender_uid": "client-1", "recipient_uid": "client-2"}
        self.assertTrue(functions["_user_chat_message_matches_pair"](own_message, "worker-1", "admin-1", pair_id))
        self.assertFalse(functions["_user_chat_message_matches_pair"](other_people_message, "worker-1", "admin-1", pair_id))
        self.assertFalse(functions["_user_chat_message_matches_pair"](own_message, "worker-1", "admin-1", "wrong-thread"))

    def test_parent_metadata_must_match_both_participants(self):
        matches = self.functions["_user_chat_parent_matches"]
        self.assertTrue(matches({"participants": ["worker-1", "admin-1"]}, "worker-1", "admin-1"))
        self.assertFalse(matches({"participants": ["worker-1", "client-1"]}, "worker-1", "admin-1"))

    def test_non_admin_cannot_direct_message_another_user(self):
        check = self.functions["_user_chat_assert_target"]
        with self.assertRaises(FakeHTTPException) as raised:
            check({"role": "user"}, {"email": "client@example.test"})
        self.assertEqual(raised.exception.status_code, 403)
        check({"role": "user"}, {"email": "typemywordz@gmail.com"})
        check({"role": "admin"}, {"email": "client@example.test"})


if __name__ == "__main__":
    unittest.main()
