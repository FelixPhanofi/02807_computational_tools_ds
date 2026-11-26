from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_audio import serve_sample


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        serve_sample(self)
