"""Petit helper d'événements envoyés au front via WebSocket.

Un événement = un dict JSON-sérialisable avec un champ ``type`` et un timestamp.
Types utilisés :
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
