import difflib
import json
import os
import unittest
from unittest.mock import MagicMock, patch

import config
from core.prompt import (
    USER_PROFILE_BLOCK,
    build_system_prompt,
    load_prompt_templates,
)
from models.ollama_manager_client import (
    FIXED_APOLOGY,
    OllamaManagerClient,
    is_clear_yes,
    is_confirmation_question,
)
from tools import system_tools
from tools.tool_registry import (
    BROWSER_ACTIONS,
    TOOL_SCHEMAS,
    TOOL_SIGNATURES,
    build_dispatch_table,
    build_mock,
    validate_tool_call,
)


class TestPromptParity(unittest.TestCase):
    def test_prompt_parity_diff_against_template(self):
        """Diffs build_system_prompt against system_prompt_template.txt."""
        tpl, tools = load_prompt_templates()
        fixed_timestamp = "2026-10-23 12:00:00"

        # In benchmark mode, memory notes are disabled
        old_bench = config.BENCHMARK_MODE
        try:
            config.BENCHMARK_MODE = True
            prompt = build_system_prompt(timestamp=fixed_timestamp)

            expected = tpl.replace("{timestamp}", fixed_timestamp).replace("{NOW}", fixed_timestamp)
            expected = expected.replace("{tool_lines}", tools.strip())
            expected = expected.replace("{tools}", tools.strip()).replace("{TOOLS}", tools.strip())
            expected = expected.replace("{FIRST}", "Akhilesh")
            if "{PROFILE}" in expected:
                expected = expected.replace("{PROFILE}", USER_PROFILE_BLOCK.strip())
            elif "{profile}" in expected:
                expected = expected.replace("{profile}", USER_PROFILE_BLOCK.strip())

            # Perform unified diff
            diff = list(
                difflib.unified_diff(
                    expected.splitlines(),
                    prompt.splitlines(),
                    fromfile="template_filled",
                    tofile="generated_prompt",
                    lineterm="",
                )
            )
            self.assertEqual(diff, [], f"Prompt diff found:\n" + "\n".join(diff))
        finally:
            config.BENCHMARK_MODE = old_bench

    def test_benchmark_mode_keeps_profile(self):
        old_bench = config.BENCHMARK_MODE
        try:
            config.BENCHMARK_MODE = True
            prompt = build_system_prompt()
            self.assertIn("Akhilesh Masali", prompt)
            self.assertIn("Asus TUF Gaming F15", prompt)
        finally:
            config.BENCHMARK_MODE = old_bench


class TestRegistryParity(unittest.TestCase):
    def test_tool_signatures_and_schemas(self):
        # All 17 tools from tool_lines.txt present
        expected_tools = {
            "open_app",
            "manage_browser",
            "play_pause_media",
            "close_current_app",
            "get_window_status",
            "send_whatsapp_message",
            "add_todo",
            "list_todos",
            "complete_todo",
            "set_timer",
            "set_alarm_at",
            "list_recent_emails",
            "create_email_draft",
            "list_upcoming_events",
            "create_event",
            "list_recent_files",
            "remember_note",
        }
        schema_names = {t["name"] for t in TOOL_SCHEMAS}
        sig_names = set(TOOL_SIGNATURES.keys())
        self.assertEqual(schema_names, expected_tools)
        self.assertEqual(sig_names, expected_tools)

        # close_current_app and others take no args
        self.assertEqual(TOOL_SIGNATURES["close_current_app"]["required"], [])
        self.assertEqual(TOOL_SIGNATURES["close_current_app"]["optional"], [])
        self.assertEqual(TOOL_SIGNATURES["play_pause_media"]["required"], [])
        self.assertEqual(TOOL_SIGNATURES["play_pause_media"]["optional"], [])
        self.assertEqual(TOOL_SIGNATURES["get_window_status"]["required"], [])
        self.assertEqual(TOOL_SIGNATURES["get_window_status"]["optional"], [])
        self.assertEqual(TOOL_SIGNATURES["list_todos"]["required"], [])
        self.assertEqual(TOOL_SIGNATURES["list_todos"]["optional"], [])

        # manage_browser actions
        expected_actions = ["new_tab", "close_tab", "reload", "back", "forward", "search", "open_url"]
        self.assertEqual(sorted(BROWSER_ACTIONS), sorted(expected_actions))
        self.assertEqual(sorted(TOOL_SIGNATURES["manage_browser"]["enums"]["action"]), sorted(expected_actions))

        # send_whatsapp_message allows contact / recipient
        self.assertIn("contact", TOOL_SIGNATURES["send_whatsapp_message"]["optional"])
        self.assertIn("recipient", TOOL_SIGNATURES["send_whatsapp_message"]["optional"])


class TestStrictDispatcher(unittest.TestCase):
    def test_unknown_tool(self):
        valid, err = validate_tool_call("unknown_tool", {})
        self.assertFalse(valid)
        self.assertTrue(err.startswith("ERROR:"))

    def test_unexpected_args_on_no_arg_tool(self):
        valid, err = validate_tool_call("close_current_app", {"window_title": "VSCode"})
        self.assertFalse(valid)
        self.assertTrue(err.startswith("ERROR: Unexpected arguments"))

    def test_missing_required_args(self):
        valid, err = validate_tool_call("open_app", {})
        self.assertFalse(valid)
        self.assertTrue(err.startswith("ERROR: Missing required arguments"))

    def test_invalid_type(self):
        valid, err = validate_tool_call("set_timer", {"seconds": "five"})
        self.assertFalse(valid)
        self.assertTrue(err.startswith("ERROR: Argument 'seconds'"))

    def test_invalid_enum(self):
        valid, err = validate_tool_call("manage_browser", {"action": "delete_all"})
        self.assertFalse(valid)
        self.assertTrue(err.startswith("ERROR: Invalid value 'delete_all'"))

    def test_valid_calls(self):
        valid, err = validate_tool_call("open_app", {"app_name": "notepad"})
        self.assertTrue(valid)
        self.assertIsNone(err)

        valid, err = validate_tool_call("manage_browser", {"action": "new_tab"})
        self.assertTrue(valid)
        self.assertIsNone(err)

        valid, err = validate_tool_call("send_whatsapp_message", {"contact": "Alice", "message": "hi"})
        self.assertTrue(valid)
        self.assertIsNone(err)

        valid, err = validate_tool_call("send_whatsapp_message", {"recipient": "Alice", "message": "hi"})
        self.assertTrue(valid)
        self.assertIsNone(err)


class TestRawJsonGuard(unittest.TestCase):
    @patch.object(OllamaManagerClient, "_call_ollama")
    def test_raw_json_invalid_retries_then_fixed_apology(self, mock_call):
        # Client returns invalid tool call twice
        mock_call.side_effect = [
            ({"message": {"content": '{"tool": "manage_browser", "args": {"action": "invalid_action"}}'}}, 10.0),
            ({"message": {"content": '{"tool": "manage_browser", "args": {"action": "still_invalid"}}'}}, 10.0),
        ]
        client = OllamaManagerClient()
        res = client.ask("Open a browser tab")
        self.assertEqual(res["reply"], FIXED_APOLOGY)
        self.assertFalse(res["reply"].startswith("{"))

    @patch.object(OllamaManagerClient, "_call_ollama")
    def test_raw_json_recovers_after_one_retry(self, mock_call):
        # Client returns invalid tool call, then recovers on retry with spoken response
        mock_call.side_effect = [
            ({"message": {"content": '{"tool": "invalid_tool", "args": {}}'}}, 10.0),
            ({"message": {"content": "I apologize. I cannot do that, but I can help you with other tasks."}}, 10.0),
        ]
        client = OllamaManagerClient()
        res = client.ask("Do something weird")
        self.assertEqual(res["reply"], "I apologize. I cannot do that, but I can help you with other tasks.")


class TestWhatsAppConfirmationGate(unittest.TestCase):
    def test_helpers(self):
        self.assertTrue(is_confirmation_question("Shall I send the WhatsApp message to Alice?"))
        self.assertTrue(is_confirmation_question("Would you like me to send this message?"))
        self.assertFalse(is_confirmation_question("Here are your unread emails."))

        self.assertTrue(is_clear_yes("yes"))
        self.assertTrue(is_clear_yes("Yes, please send it now."))
        self.assertTrue(is_clear_yes("Sure, go ahead"))
        self.assertFalse(is_clear_yes("No, cancel it"))
        self.assertFalse(is_clear_yes("Why would I do that?"))

    @patch.object(OllamaManagerClient, "_call_ollama")
    def test_refuse_without_prior_confirmation_question(self, mock_call):
        mock_call.return_value = (
            {"message": {"content": '{"tool": "send_whatsapp_message", "args": {"recipient": "Alice", "message": "hello"}}'}},
            10.0,
        )
        client = OllamaManagerClient()
        client.chat_history = [
            {"role": "user", "content": "Hello Jarvis"},
            {"role": "assistant", "content": "Hello sir, how can I assist you today?"},
        ]
        # Current user message is yes, but assistant didn't ask confirmation question
        res = client.ask("yes")
        # Tool was refused
        self.assertIn("User confirmation required", res.get("error", ""))

    @patch.object(OllamaManagerClient, "_call_ollama")
    def test_refuse_without_user_clear_yes(self, mock_call):
        mock_call.return_value = (
            {"message": {"content": '{"tool": "send_whatsapp_message", "args": {"recipient": "Alice", "message": "hello"}}'}},
            10.0,
        )
        client = OllamaManagerClient()
        client.chat_history = [
            {"role": "user", "content": "Send a message to Alice on WhatsApp"},
            {"role": "assistant", "content": "Shall I send 'hello' to Alice on WhatsApp?"},
        ]
        res = client.ask("Wait, don't send it yet")
        self.assertIn("User confirmation required", res.get("error", ""))

    @patch.object(OllamaManagerClient, "_call_ollama")
    def test_accept_with_prior_confirmation_and_user_yes(self, mock_call):
        mock_call.side_effect = [
            ({"message": {"content": '{"tool": "send_whatsapp_message", "args": {"contact": "Alice", "message": "hello"}}'}}, 10.0),
            ({"message": {"content": "I have sent the message to Alice."}}, 10.0),
        ]
        client = OllamaManagerClient()
        client.chat_history = [
            {"role": "user", "content": "Send a message to Alice"},
            {"role": "assistant", "content": "Shall I send the WhatsApp message to Alice?"},
        ]
        res = client.ask("Yes, please send it now.", force_mock=True)
        self.assertEqual(res["reply"], "I have sent the message to Alice.")
        self.assertEqual(res["tool_called"], "send_whatsapp_message")


class TestOpenAppUnknownError(unittest.TestCase):
    def test_open_app_unknown(self):
        res = system_tools.open_app("nonexistent_app_xyz_123")
        self.assertEqual(res, "App not found on system.")

    def test_dispatch_table_mock_returns_real_error(self):
        mock = build_mock()
        table = build_dispatch_table(mock=mock)
        res = table["open_app"]("nonexistent_app_xyz_123")
        self.assertEqual(res, "App not found on system.")


if __name__ == "__main__":
    unittest.main()
