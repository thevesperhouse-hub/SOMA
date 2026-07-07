"""Download a model file from a URL (HF / Civitai / any direct link) onto the machine,
streaming with live progress — so cloud users can grab their base model without scp.

Runs in a thread like the caption/train jobs and emits `model_fetch` events
({state: downloading|done|error|stopped, percent, mb, total_mb, path, name}).
"""
import os
import threading
import time
import urllib.parse
import urllib.request

from events import evt


def _filename(resp, url) -> str:
    cd = resp.headers.get("Content-Disposition", "") or ""
    if "filename=" in cd:
        n = cd.split("filename=")[-1].strip().strip('";')
        n = os.path.basename(urllib.parse.unquote(n))
        if n:
            return n
    n = os.path.basename(urllib.parse.urlparse(url).path)
    return n or "model.safetensors"


class ModelFetchJob(threading.Thread):
    def __init__(self, url, dest_dir, emit):
        super().__init__(daemon=True)
        self.url = url
        self.dest_dir = dest_dir
        self.emit = emit
        self._stop = threading.Event()
        self.result_path = None

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self._fetch()
        except Exception as e:
            self.emit(evt("model_fetch", state="error", message=str(e)))

    def _fetch(self):
        os.makedirs(self.dest_dir, exist_ok=True)
        url = self.url
        # Civitai auth: add the global token (SOMA_CIVITAI_TOKEN) if the URL is a Civitai
        # download without one — many models require a logged-in account.
        civ_tok = os.environ.get("SOMA_CIVITAI_TOKEN", "").strip()
        if civ_tok and "civitai." in url and "token=" not in url:
            url += ("&" if "?" in url else "?") + "token=" + civ_tok
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        self.emit(evt("model_fetch", state="downloading", percent=0))
        with urllib.request.urlopen(req, timeout=30) as resp:
            # Auth wall: many hosts (Civitai) 3xx-redirect to a login/HTML page instead of
            # the file. Detect it and fail loudly instead of saving the login page.
            final = resp.geturl()
            ctype = (resp.headers.get("Content-Type", "") or "").lower()
            if "/login" in final or "text/html" in ctype:
                self.emit(evt("model_fetch", state="error",
                              message="Auth required — this model needs a valid Civitai token "
                                      "(add ?token=YOUR_KEY, or set SOMA_CIVITAI_TOKEN)"))
                return
            name = _filename(resp, url)
            total = int(resp.headers.get("Content-Length", 0) or 0)
            tmp = os.path.join(self.dest_dir, name + ".part")
            done = 0
            last = -1
            with open(tmp, "wb") as f:
                while True:
                    if self._stop.is_set():
                        self.emit(evt("model_fetch", state="stopped"))
                        return
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = int(done * 100 / total)
                        if pct != last:
                            last = pct
                            self.emit(evt("model_fetch", state="downloading", percent=pct,
                                          mb=round(done / 1e6), total_mb=round(total / 1e6)))
        path = os.path.join(self.dest_dir, name)
        os.replace(tmp, path)  # atomic: only appears complete when fully downloaded
        self.result_path = path
        self.emit(evt("model_fetch", state="done", path=path, name=name,
                      mb=round(done / 1e6)))
