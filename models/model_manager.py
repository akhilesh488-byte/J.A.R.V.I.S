import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.claude_client import ClaudeClient
from models.openai_client import OpenAIClient
from models.gemini_client import GeminiClient


class ModelManager:
    def __init__(self):
        self._clients = {
            "claude": ClaudeClient(),
            "openai": OpenAIClient(),
            "gemini": GeminiClient(),
        }

    def get_client(self, name: str):
        return self._clients.get(name)

    def resolve(self, requested: str = None):
        """
        Returns the ModelClient to use.
        If `requested` names a model (or alias) that's available, use it.
        Otherwise walk MODEL_PRIORITY and return the first available one.
        Returns None if nothing is configured yet.
        """
        if requested:
            key = config.MODEL_ALIASES.get(requested.strip().lower())
            if key:
                client = self._clients.get(key)
                if client and client.is_available():
                    return client

        for key in config.MODEL_PRIORITY:
            client = self._clients.get(key)
            if client and client.is_available():
                return client

        return None
