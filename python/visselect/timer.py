# --- Imports ------------------------------------------------------------------

# standard library
import time


# --- Timer Class --------------------------------------------------------------

class Timer:
    """
    A stopwatch that accumulates the computation time of a solve across runs
    """

    def __init__(self) -> None:
        """Create a new stopped timer with no accumulated time"""
        self._elapsed = 0.0
        self._start = None

    def __enter__(self) -> "Timer":
        """Start the timer on entering a with block"""
        self.start()
        return self

    def __exit__(self, *exception) -> None:
        """Stop the timer on leaving a with block, even on an exception"""
        self.stop()

    def start(self) -> None:
        """
        Start timing, adding to any previously accumulated time

        Raises:
            RuntimeError: If the timer is already running
        """
        if self._start is not None:
            raise RuntimeError("Timer is already running")
        self._start = time.perf_counter()

    def stop(self) -> None:
        """
        Stop timing and add the time since start to the accumulated time

        Raises:
            RuntimeError: If the timer is not running
        """
        if self._start is None:
            raise RuntimeError("Timer is not running")
        self._elapsed += time.perf_counter() - self._start
        self._start = None

    @property
    def elapsed(self) -> float:
        """
        The accumulated time in seconds, including the current run if running
        """
        if self._start is None:
            return self._elapsed
        return self._elapsed + time.perf_counter() - self._start

    @property
    def running(self) -> bool:
        """
        Whether the timer is currently running
        """
        return self._start is not None
