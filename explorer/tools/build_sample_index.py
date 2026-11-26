"""Locate playable original recordings by byte range in the official FMA Medium ZIP."""
import hashlib
import json
import struct
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main():
    atlas = json.loads((HERE/"public/search-atlas/songs.json").read_text(encoding="utf8"))
    samples = {r["track_id"] for r in json.loads((HERE/"public/listening-catalogue.json").read_text(encoding="utf8"))}
    entries = {}
    archive = HERE.parent/"data/medium/data/raw/fma_medium.zip"
    with zipfile.ZipFile(archive) as package, archive.open("rb") as source:
        for tid, _, _, _, _, _, _, license_idx in atlas["songs"]:
            if not atlas["licenses"][license_idx][2]:
                continue
            name = f"{tid:06d}"
            info = package.getinfo(f"fma_medium/{name[:3]}/{name}.mp3")
            assert info.compress_type == zipfile.ZIP_BZIP2 and info.file_size < 4_000_000
            source.seek(info.header_offset)
            header = source.read(30)
            assert header[:4] == b"PK\x03\x04"
            filename, extra = struct.unpack_from("<HH",header,26)
            entry = dict(start=info.header_offset+30+filename+extra, compressed=info.compress_size,
                         size=info.file_size, crc=info.CRC)
            if tid in samples:
                entry["sha256"] = hashlib.sha256(package.read(info)).hexdigest()
            entries[str(tid)] = entry
    (HERE/"sample-index.json").write_text(json.dumps(entries,separators=(",",":"))+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
