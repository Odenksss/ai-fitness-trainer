import threading
import queue
import time


class VoiceFeedback:
    """
    Voice feedback using Windows SAPI via win32com.
    Runs a dedicated speaker thread with a message queue.
    Keeps talking until you press q.
    """

    def __init__(self, cooldown=2.0, rate=150):
        self.cooldown = cooldown
        self._last_spoken: dict[str, float] = {}
        self._queue = queue.Queue()

        # Start a single dedicated speaker thread
        t = threading.Thread(target=self._speaker_loop, daemon=True)
        t.start()

    def _speaker_loop(self):
        """Dedicated thread that owns the SAPI voice object."""
        import win32com.client
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        while True:
            message = self._queue.get()   # blocks until something arrives
            try:
                speaker.Speak(message)    # synchronous — finishes before next
            except Exception:
                pass

    def speak(self, message: str, force=False):
        now = time.time()
        if not force:
            if now - self._last_spoken.get(message, 0) < self.cooldown:
                return
        self._last_spoken[message] = now
        self._queue.put(message)

    def speak_errors(self, errors: list):
        for err in errors:
            self.speak(err)

    def speak_rep(self, count: int):
        self.speak(f"Rep {count}", force=True)
