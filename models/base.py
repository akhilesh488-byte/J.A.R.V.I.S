"""
Every model client implements the same tiny interface so the router
and model_manager don't need to know which provider they're talking to.
"""

from abc import ABC, abstractmethod


class ModelClient(ABC):
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this client has what it needs (e.g. an API key) to run."""
        raise NotImplementedError

    @abstractmethod
    def ask(self, prompt: str, dispatch_table: dict, image_path: str = None) -> str:
        """
        Send `prompt` to the model. If the model decides to use a tool
        (from dispatch_table), execute it and feed the result back,
        looping until the model gives a final text answer.
        If image_path is given, attach it (for "what's on my screen" style
        questions) — implementers that don't support vision can ignore it.
        Returns the final text response to speak back to the user.
        """
        raise NotImplementedError
