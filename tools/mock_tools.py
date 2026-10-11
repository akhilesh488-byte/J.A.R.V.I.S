"""
Mock tools for benchmarking. Every tool just records (name, args) and
returns a fixed string, so benchmark runs never open apps, touch the
to-do list, or create calendar events.
"""

import time


class MockTools:
    def __init__(self, names):
        self.names = list(names)
        self.calls = []

    def reset(self):
        self.calls = []

    def table(self):
        def make(name):
            def fn(**kwargs):
                self.calls.append({"tool": name, "args": kwargs, "t": time.perf_counter()})
                return f"ok: {name} (mock)"
            return fn
        return {n: make(n) for n in self.names}
