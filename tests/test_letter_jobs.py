"""Regression coverage for unsplit Letter Jobs and stable worker-audio encoding."""
import ast
import base64
import hashlib
import os
import re
import shutil
import unittest
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

try:
    from pydub import AudioSegment
    from pydub.generators import Sine
    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None
    Sine = None
    PYDUB_AVAILABLE = False

MAIN_PATH = next((parent / 'main.py' for parent in Path(__file__).resolve().parents if (parent / 'main.py').exists()), Path(__file__).resolve().parent / 'main.py')


def load_helpers():
    source = MAIN_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source)
    wanted_functions = {
        '_human_template_docx_profile', '_human_template_line_kind', '_human_template_source_kind',
        '_human_template_render_docx', '_human_letter_template_bytes', '_human_letter_guidelines',
        '_human_letter_render_docx', '_human_letter_docx_text', '_human_worker_audio_cache_key',
        '_human_worker_playback_mp3_bytes', '_human_worker_split_mp3_bytes', '_human_ai_agent_system',
    }
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted_functions:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(target, 'id', '') == 'HUMAN_AI_AGENTS' for target in node.targets):
            body.append(node)
    namespace = {
        'base64': base64, 'hashlib': hashlib, 'os': os, 're': re, 'datetime': datetime,
        'BytesIO': BytesIO, 'Document': Document, 'OxmlElement': OxmlElement,
        'Paragraph': Paragraph, 'deepcopy': deepcopy, 'qn': qn,
        'AudioSegment': AudioSegment, '__file__': str(MAIN_PATH),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(MAIN_PATH), 'exec'), namespace)
    return namespace, source


class LetterJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, cls.source = load_helpers()

    def test_bundled_template_is_usable_and_has_no_tables(self):
        raw = self.ns['_human_letter_template_bytes']()
        document = Document(BytesIO(raw))
        self.assertEqual(document.tables, [])
        self.assertIn('Date', self.ns['_human_template_docx_profile'](raw))
        self.assertIn('PERMANENT GUIDELINES FOR THE DEDICATED TYPEMYWORDZ LETTER AGENT', self.ns['_human_letter_guidelines']())

    def test_letter_agent_has_an_explicit_correspondence_override_and_opus_first_models(self):
        agent = self.ns['HUMAN_AI_AGENTS']['letter-opus']
        self.assertEqual(agent['job_types'], ['letter_job'])
        self.assertEqual(agent['models'], ['gpt-5.6-sol', 'claude-opus-5-5'])
        system = self.ns['_human_ai_agent_system']('letter-opus', 'formatting', '', '')
        self.assertIn('dedicated LETTER JOB', system)
        self.assertIn('never refuse it', system)
        self.assertIn('PERMANENT LETTER-JOB GUIDELINES', system)
        self.assertNotIn('If the source appears to be correspondence, return exactly TEMPLATE_JOB_BLOCKED_LETTER', system)

    def test_rendered_docx_keeps_required_fields_and_bolds_only_staff_instructions(self):
        template = self.ns['_human_letter_template_bytes']()
        draft = (
            'Date: October 5, 2026\n\n\nRe: Lucy\n\nDear :\n\n'
            '\tThis is one dictated paragraph.\n\n[Please send this to Lucy.]\n\n'
            'Client spellings: None; My spellings: Lucy'
        )
        rendered = self.ns['_human_letter_render_docx'](template, draft)
        document = Document(BytesIO(rendered))
        visible = [paragraph for paragraph in document.paragraphs if paragraph.text.strip()]
        text = '\n'.join(paragraph.text for paragraph in visible)
        self.assertIn('Date: October 5, 2026', text)
        self.assertIn('Re: Lucy', text)
        self.assertIn('Dear :', text)
        self.assertNotIn('File No.:', text)
        self.assertNotIn('Claim Date:', text)
        self.assertNotIn('Be sure to delete any unused template information', text)
        staff = next(paragraph for paragraph in visible if paragraph.text.startswith('['))
        spellings = next(paragraph for paragraph in visible if paragraph.text.startswith('Client spellings:'))
        self.assertTrue(all(run.bold is True for run in staff.runs))
        self.assertTrue(all(run.bold is False for run in spellings.runs))
        self.assertEqual(self.ns['_human_letter_docx_text'](rendered).splitlines()[0], 'Date: October 5, 2026')

    def test_worker_audio_cache_key_changes_for_replaced_source_or_range(self):
        key = self.ns['_human_worker_audio_cache_key']
        source = key('human-workflow/job/audio/source-a.mp3', 1000, 'part-1', 0, 30)
        replacement = key('human-workflow/job/audio/source-b.mp3', 1000, 'part-1', 0, 30)
        other_range = key('human-workflow/job/audio/source-a.mp3', 1000, 'part-1', 30, 60)
        self.assertNotEqual(source, replacement)
        self.assertNotEqual(source, other_range)
        self.assertEqual(len(source), 20)

    def test_generic_template_agent_still_refuses_correspondence(self):
        generic_guidelines = (MAIN_PATH.parent / 'template_agent_guidelines.txt').read_text(encoding='utf-8')
        self.assertIn('return exactly `TEMPLATE_JOB_BLOCKED_LETTER`', generic_guidelines)
        self.assertIn('If the source appears to be correspondence, return exactly TEMPLATE_JOB_BLOCKED_LETTER', self.source)

    def test_letter_submission_and_review_are_server_gated(self):
        self.assertIn('Attach the finished Word document (.docx) before submitting a Letter Job.', self.source)
        self.assertIn('Letter Jobs must remain one complete, unsplit assignment.', self.source)
        self.assertIn('Run AI proofreading for this Letter Job or assign a human proofreader before final admin approval.', self.source)
        self.assertIn('Assign this complete Letter Job from the Letter Jobs section', self.source)


@unittest.skipUnless(PYDUB_AVAILABLE and shutil.which('ffmpeg'), 'pydub and ffmpeg are required for audio codec tests')
class WorkerAudioEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, _ = load_helpers()

    def test_normalized_mono_mp3_is_audible_and_standard_rate(self):
        source = Sine(440).to_audio_segment(duration=1800).set_channels(1).set_frame_rate(48000)
        output = self.ns['_human_worker_playback_mp3_bytes'](source)
        decoded = AudioSegment.from_file(BytesIO(output), format='mp3')
        self.assertEqual(decoded.frame_rate, 44100)
        self.assertGreater(decoded.rms, 1000)
        self.assertLess(abs(len(decoded) - len(source)), 150)

    def test_distinct_stereo_tracks_remain_separate_and_audible(self):
        left = Sine(440).to_audio_segment(duration=1800).set_channels(1)
        right = Sine(880).to_audio_segment(duration=1800).set_channels(1)
        source = AudioSegment.from_mono_audiosegments(left, right)
        output = self.ns['_human_worker_split_mp3_bytes'](source)
        decoded = AudioSegment.from_file(BytesIO(output), format='mp3')
        channels = decoded.split_to_mono()
        self.assertEqual(decoded.frame_rate, 44100)
        self.assertEqual(len(channels), 2)
        self.assertTrue(all(channel.rms > 500 for channel in channels))
        self.assertLess(abs(len(decoded) - len(source)), 150)


if __name__ == '__main__':
    unittest.main()
