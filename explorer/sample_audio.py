"""Retrieve only catalogued recordings from the official FMA Medium archive."""
import bz2
import hashlib
import json
import zlib
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

ARCHIVE = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
TRACKS = json.loads(Path(__file__).with_name("sample-index.json").read_text(encoding="utf8"))


def serve_sample(handler):
    values = parse_qs(urlsplit(handler.path).query).get("id", [])
    if len(values) != 1 or values[0] not in TRACKS:
        return handler.send_error(404, "Unknown sample")
    track = TRACKS[values[0]]
    start = track["start"]
    end = start + track["compressed"] - 1
    # A per-track cache key keeps intermediary caches from mixing different partial responses.
    request = Request(f"{ARCHIVE}?sample={values[0]}", headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    try:
        with urlopen(request, timeout=18) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                raise ValueError("Archive did not honor the byte range")
            compressed = response.read(track["compressed"] + 1)
        if len(compressed) != track["compressed"]:
            raise ValueError("Incomplete sample")
        decoder = bz2.BZ2Decompressor()
        audio = decoder.decompress(compressed, max_length=track["size"] + 1)
        if not decoder.eof or len(audio) != track["size"] or zlib.crc32(audio) != track["crc"]:
            raise ValueError("Sample failed archive checksum")
        if "sha256" in track and hashlib.sha256(audio).hexdigest() != track["sha256"]:
            raise ValueError("Sample failed verification")
    except (OSError, ValueError, EOFError):
        return handler.send_error(502, "This recording could not be loaded")
    handler.send_response(200)
    handler.send_header("Content-Type", "audio/mpeg")
    handler.send_header("Content-Length", str(len(audio)))
    handler.send_header("Cache-Control", "public, max-age=86400, s-maxage=604800")
    handler.end_headers()
    handler.wfile.write(audio)
