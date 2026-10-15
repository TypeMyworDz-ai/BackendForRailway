import asyncio
import gzip
import json
import pathlib
import re
import unittest

SRC = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text()
START = SRC.index("class _JsonGzipMiddleware:")
END = SRC.index("app.add_middleware(_JsonGzipMiddleware)")
NS = {}
exec(SRC[START:END], NS)
Mw = NS["_JsonGzipMiddleware"]


def run(app_messages, headers=None, method="GET"):
    async def app(scope, receive, send):
        for m in app_messages:
            await send(m)
    sent = []

    async def send(m):
        sent.append(m)

    async def receive():
        return {"type": "http.request"}
    scope = {"type": "http", "method": method, "headers": headers if headers is not None else [(b"accept-encoding", b"gzip, br")]}
    asyncio.run(Mw(app)(scope, receive, send))
    return sent


def resp(ctype=b"application/json", body=b"", status=200, extra=()):
    return [
        {"type": "http.response.start", "status": status, "headers": [(b"content-type", ctype), (b"content-length", str(len(body)).encode()), *extra]},
        {"type": "http.response.body", "body": body},
    ]


BIG = json.dumps({"rows": ["abcdefghij" * 5] * 200}).encode()


class GzipTests(unittest.TestCase):
    def test_big_json_is_compressed(self):
        sent = run(resp(body=BIG))
        h = dict(sent[0]["headers"])
        self.assertEqual(h[b"content-encoding"], b"gzip")
        self.assertEqual(gzip.decompress(sent[1]["body"]), BIG)
        self.assertEqual(int(h[b"content-length"]), len(sent[1]["body"]))
        self.assertIn(b"accept-encoding", h[b"vary"].lower())

    def test_small_json_untouched(self):
        sent = run(resp(body=b'{"ok":true}'))
        self.assertNotIn(b"content-encoding", dict(sent[0]["headers"]))
        self.assertEqual(sent[1]["body"], b'{"ok":true}')

    def test_audio_pdf_docx_stream_untouched(self):
        for ctype in (b"audio/mpeg", b"application/pdf", b"text/event-stream",
                      b"application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"text/html"):
            sent = run(resp(ctype=ctype, body=BIG))
            self.assertNotIn(b"content-encoding", dict(sent[0]["headers"]), ctype)
            self.assertEqual(sent[1]["body"], BIG)

    def test_range_head_and_no_gzip_header_untouched(self):
        for headers, method in (([(b"accept-encoding", b"gzip"), (b"range", b"bytes=0-")], "GET"),
                                ([(b"accept-encoding", b"gzip")], "HEAD"),
                                ([], "GET")):
            sent = run(resp(body=BIG), headers=headers, method=method)
            self.assertNotIn(b"content-encoding", dict(sent[0]["headers"]))
            self.assertEqual(sent[1]["body"], BIG)

    def test_errors_partial_and_encoded_untouched(self):
        for r in (resp(body=BIG, status=404), resp(body=BIG, status=206),
                  resp(body=BIG, extra=[(b"content-encoding", b"br")])):
            sent = run(r)
            self.assertEqual(sent[1]["body"], BIG)

    def test_streaming_json_passes_through(self):
        msgs = [resp(body=b"")[0],
                {"type": "http.response.body", "body": b"x" * 2000, "more_body": True},
                {"type": "http.response.body", "body": b"y" * 2000, "more_body": False}]
        sent = run(msgs)
        self.assertNotIn(b"content-encoding", dict(sent[0]["headers"]))
        self.assertEqual(b"".join(m["body"] for m in sent[1:]), b"x" * 2000 + b"y" * 2000)

    def test_registered_once(self):
        self.assertEqual(SRC.count("app.add_middleware(_JsonGzipMiddleware)"), 1)


if __name__ == "__main__":
    unittest.main()
