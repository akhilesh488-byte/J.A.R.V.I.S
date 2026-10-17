import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.claude_client import ClaudeClient
from models.gemini_client import GeminiClient
from models.openai_client import LocalClient, OpenAIClient
from models.ollama_manager_client import OllamaManagerClient


class ModelManager:
    def __init__(self):
        self._clients = {
            "claude": ClaudeClient(),
            "openai": OpenAIClient(),
            "gemini": GeminiClient(),
            "local": LocalClient(),
            "ollama": OllamaManagerClient(model_manager=self),
            "ollama_manager": OllamaManagerClient(model_manager=self),
        }

    def get_client(self, name: str):
        return self._clients.get(name)

    def available(self):
        return [k for k, c in self._clients.items() if c.is_available()]

    def candidates(self, requested: str = None):
        """
        Available clients in the order to try them: an explicitly requested
        model first (if it's available), then MODEL_PRIORITY.
        """
        order = []
        if requested:
            key = config.MODEL_ALIASES.get(requested.strip().lower(), requested.strip().lower())
            if key:
                order.append(key)

        env_provider = os.environ.get("MODEL_PROVIDER", "").strip().lower()
        if env_provider in ("ollama", "local"):
            if "ollama_manager" not in order:
                order.append("ollama_manager")
            if "ollama" not in order:
                order.append("ollama")

        order += [k for k in config.MODEL_PRIORITY if k not in order]
        if "ollama_manager" not in order:
            order.append("ollama_manager")

        return [self._clients[k] for k in order
                if k in self._clients and self._clients[k].is_available()]

    def resolve(self, requested: str = None):
        c = self.candidates(requested)
        return c[0] if c else None
