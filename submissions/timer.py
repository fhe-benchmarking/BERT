import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self._start = time.perf_counter()
        self.paused_time = 0.0

    def pause(self):
        self._pause_start = time.perf_counter()

    def resume(self):
        self.paused_time += time.perf_counter() - self._pause_start

    @contextmanager
    def paused(self):
        self.pause()
        try:
            yield
        finally:
            self.resume()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    @property
    def compute_elapsed(self) -> float:
        return self.elapsed - self.paused_time
