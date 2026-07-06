"""Small helper for events sent to the front-end over WebSocket.

An event = a JSON-serializable dict with a ``type`` field and a timestamp.
Types used:
  - status : {state: starting|training|sampling|done|stopped|error, ...}
  - step   : {step, total_steps, loss, lr, secs}
  - sample : {step, total_steps, placeholder, image?|seed, prompt, sharpness}
  - log    : {level, message}
"""
import time


def evt(type_: str, **kw) -> dict:
    e = {"type": type_, "t": round(time.time(), 3)}
    e.update(kw)
    return e
