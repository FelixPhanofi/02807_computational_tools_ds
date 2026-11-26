"""Public recognition endpoint with a bounded wait for a busy GPU and a landmark fallback."""
from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_match import identify


def gpu_answer(request):
    deadline = time.monotonic() + 210
    for delay in (0, 1, 2):
        remaining = deadline - time.monotonic() - delay
        if remaining < 1:
            break
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=min(205, remaining)) as response:
                answer = json.load(response)
            return answer if isinstance(answer, dict) and isinstance(answer.get("accepted"), bool) else None
        except urllib.error.HTTPError as error:
            error.close()
            if error.code != 429:
                break
        except (OSError, ValueError):
            break
    return None


class handler(BaseHTTPRequestHandler):
    """Accept only a ten-second standardised PCM payload."""

    def reply(self, status, obj):
        """Return JSON without storing audio or caching individual responses."""
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        """Score the submitted waveform on the available backend."""
        if self.headers.get_content_type() != "application/octet-stream":
            return self.reply(415, {"error": "Expected signed mono PCM at 24 kHz."})
        if self.headers.get("Content-Length") != "480000":
            return self.reply(413, {"error": "Send exactly ten seconds of audio."})
        raw = self.rfile.read(480000)
        if len(raw) != 480000:
            return self.reply(400, {"error": "Incomplete audio."})
        url = os.environ.get("SIGNAL_GPU_URL", "")
        token = os.environ.get("SIGNAL_GPU_TOKEN", "")
        if url.startswith("https://") and token:
            request = urllib.request.Request(
                url, data=raw, headers={"Content-Type": "application/octet-stream", "Authorization": "Bearer " + token}
            )
            answer = gpu_answer(request)
            if answer is not None:
                if urllib.parse.urlsplit(url).hostname.endswith(".azurecontainerapps.io"):
                    answer["engine"] = f"Azure · {answer['engine']}"
                return self.reply(200, answer)
        return self.reply(200, identify(raw))
