"""Loopback-only recording explorer. An optional second listener serves authenticated inference only."""
import argparse
import functools
import hmac
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import traceback

HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    """Serve public assets and accept only one bounded PCM query at a time."""

    def reply(self,status,body):
        """Send a bounded JSON response."""
        raw=json.dumps(body,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(raw)))
        self.send_header("Cache-Control","no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        """Use a health probe without loading or querying the model."""
        if self.server.inference_only:
            return self.reply(404,{})
        if self.path == "/api/health":
            return self.reply(200,dict(recognition=True,engine="MuQ + LSH · FMA Medium"))
        if self.path.partition("?")[0] == "/api/sample":
            from sample_audio import serve_sample
            return serve_sample(self)
        return super().do_GET()

    def do_POST(self):
        """Validate the request before allocating model resources."""
        if self.path != "/api/recognize":
            return self.reply(404,{})
        if self.server.inference_only and not hmac.compare_digest(self.headers.get("Authorization",""),"Bearer "+self.server.key):
            return self.reply(401,{"error":"Unauthorized"})
        if self.headers.get_content_type()!="application/octet-stream":
            return self.reply(415,{"error":"Expected mono PCM16 at 24 kHz."})
        if self.headers.get("Content-Length")!="480000":
            return self.reply(413,{"error":"Expected exactly ten seconds of audio."})
        if not LOCK.acquire(False):
            return self.reply(429,{"error":"The GPU is processing another clip. Try again."})
        try:
            self.connection.settimeout(12)
            raw=self.rfile.read(480000)
            if len(raw)!=480000:
                return self.reply(400,{"error":"Incomplete recording."})
            self.reply(200,self.server.model.identify(raw))
        except Exception:
            traceback.print_exc()
            self.reply(503,{"error":"GPU recognition is temporarily unavailable."})
        finally:
            LOCK.release()

    def do_HEAD(self):
        """Keep the relay listener limited to the recognition endpoint."""
        if self.server.inference_only:
            return self.reply(404,{})
        super().do_HEAD()


if __name__=="__main__":
    from gpu_match import Recognizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8327)
    parser.add_argument("--cloud", action="store_true", help="Serve authenticated inference on the container interface.")
    args=parser.parse_args()
    if args.cloud and not os.environ.get("SIGNAL_RELAY_TOKEN"):
        parser.error("Cloud mode requires SIGNAL_RELAY_TOKEN.")
    model=Recognizer()
    def server(port,relay):
        host = "0.0.0.0" if args.cloud else "127.0.0.1"
        service=ThreadingHTTPServer((host,port),functools.partial(Handler,directory=str(HERE/"public")))
        service.model=model
        service.inference_only=relay
        service.key=os.environ.get("SIGNAL_RELAY_TOKEN","")
        return service
    if not args.cloud and os.environ.get("SIGNAL_RELAY_TOKEN"):
        threading.Thread(target=server(args.port+1,True).serve_forever,daemon=True).start()
    server(args.port,args.cloud).serve_forever()

