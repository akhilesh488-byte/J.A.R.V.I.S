import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jarvis_local
from jarvis_local import (
    JarvisPersonalityRouter,
    clean_text_for_tts,
    get_system_telemetry,
    get_time_based_greeting,
    voice_synth,
)


class TestJarvisEnhancedFeatures(unittest.TestCase):
    def setUp(self):
        self.spoken = []
        jarvis_local.speak = lambda text: self.spoken.append(text)

    def test_clean_text_for_tts(self):
        dirty = "System initialized! *Running* at 100% #speed... Ready? 😀"
        cleaned = clean_text_for_tts(dirty)
        self.assertNotIn("*", cleaned)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("?", cleaned)
        self.assertTrue(cleaned.endswith("."))

    def test_dynamic_greeting(self):
        greeting = get_time_based_greeting()
        self.assertTrue(any(p in greeting for p in ["Good morning, sir.", "Good afternoon, sir.", "Good evening, sir."]))
        self.assertIn("All primary systems are online and functioning within nominal parameters", greeting)
        self.assertIn("How may I assist you today, sir", greeting)

    def test_introduce_yourself(self):
        handled = JarvisPersonalityRouter.execute("introduce yourself")
        self.assertTrue(handled)
        self.assertTrue(any("Allow me to introduce myself" in s for s in self.spoken))
        self.assertTrue(any("All core modules are at your disposal, sir." in s for s in self.spoken))

        # Test alternate phrasing
        self.spoken.clear()
        self.assertTrue(JarvisPersonalityRouter.execute("who are you"))
        self.assertTrue(any("Allow me to introduce myself" in s for s in self.spoken))

    def test_hardware_telemetry(self):
        data = get_system_telemetry()
        self.assertIn("cpu_load", data)
        self.assertIn("mem_percent", data)
        self.assertIn("gpu_temp", data)
        self.assertIn("power_mode", data)
        self.assertIsInstance(data["cpu_load"], int)
        self.assertIsInstance(data["mem_percent"], int)

        # Test voice command execution
        self.spoken.clear()
        handled = JarvisPersonalityRouter.execute("system status")
        self.assertTrue(handled)
        self.assertTrue(any("Hardware telemetry report, sir." in s for s in self.spoken))
        self.assertTrue(any("All thermal parameters are nominal, sir." in s for s in self.spoken))

    def test_single_reference_audio(self):
        ref = voice_synth.reference_wav
        self.assertIsNotNone(ref)
        self.assertIsInstance(ref, str)
        self.assertTrue(os.path.exists(ref))
        self.assertTrue(ref.endswith(".wav"))

    def test_youtube_playback(self):
        handled = JarvisPersonalityRouter.execute("play interstellar theme on youtube")
        self.assertTrue(handled)
        self.assertTrue(any("Searching YouTube and initiating playback for Interstellar Theme, sir." in s for s in self.spoken))

    def test_spotify_playback(self):
        handled = JarvisPersonalityRouter.execute("play starboy on spotify")
        self.assertTrue(handled)
        self.assertTrue(any("Searching Spotify and loading Starboy, sir." in s for s in self.spoken))

    def test_google_search(self):
        handled = JarvisPersonalityRouter.execute("search artificial intelligence on google")
        self.assertTrue(handled)
        self.assertTrue(any("Executing Google search for artificial intelligence, sir." in s for s in self.spoken))

    def test_todo_google_tasks(self):
        handled = JarvisPersonalityRouter.execute("open to do list")
        self.assertTrue(handled)
        self.assertTrue(any("Opening Google Tasks to manage your to-do list, sir." in s for s in self.spoken))

    def test_media_stop(self):
        handled = JarvisPersonalityRouter.execute("stop the song")
        self.assertTrue(handled)
        self.assertTrue(any("Playback stopped, sir." in s for s in self.spoken))

    def test_app_launching(self):
        handled = JarvisPersonalityRouter.execute("open calculator")
        self.assertTrue(handled)
        self.assertTrue(any("Launching Calculator now, sir." in s for s in self.spoken))

    def test_time_and_date(self):
        self.spoken.clear()
        self.assertTrue(JarvisPersonalityRouter.execute("what time is it"))
        self.assertTrue(any("The current time is" in s for s in self.spoken))

        self.spoken.clear()
        self.assertTrue(JarvisPersonalityRouter.execute("what is today's date"))
        self.assertTrue(any("Today is" in s for s in self.spoken))

    def test_unknown_command(self):
        handled = JarvisPersonalityRouter.execute("make me a pizza")
        self.assertFalse(handled)
        self.assertTrue(any("Command unrecognized, sir. Standing by." in s for s in self.spoken))


if __name__ == "__main__":
    unittest.main()
