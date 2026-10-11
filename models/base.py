"""
Every model client implements the same tiny interface, plus a
`last_stats` dict filled in by each ask() call (used by the benchmark
and by the router's logging).
"""

from abc import ABC, abstractmethod


def new_stats(model_id=""):
    return {
        "model": model_id,
        "latency_ms": 0.0,      # time spent inside API calls only (not tools)
        "tokens_in": 0,
        "tokens_out": 0,
        "iterations": 0,
        "tool_calls": [],       # [{"tool": name, "args": {...}}] in call order
    }


class ModelClient(ABC):
    name: str = "base"

    def __init__(self):
        self.last_stats = new_stats()

    @abstractmethod
    def is_available(self) -> bool:
        """True if this client has what it needs (API key, local model name)."""
        raise NotImplementedError

    @abstractmethod
    def ask(self, prompt: str, dispatch_table: dict, image_path: str = None) -> str:
        """
        Send `prompt` to the model, run any tool calls it makes through
        dispatch_table, loop until it gives a final answer, return the text.
        Fills self.last_stats. Raises on API errors (the router handles them).
        """
        raise NotImplementedError
