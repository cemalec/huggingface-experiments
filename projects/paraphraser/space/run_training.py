"""
Space entry point.

Starts an HTTP server on :7860 immediately (satisfies HF's 30-min health check),
then runs training in the main thread. After a successful push the Space pauses
itself via the HF API so billing stops automatically.
"""
import http.server
import os
import socketserver
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

FLAG = Path("/tmp/.training_done")
SPACE_ID = "cemalec/paraphraser-training"


def start_health_server(status: dict):
    """Serve :7860 in a daemon thread so HF's health check passes immediately."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            msg = status.get("message", "Starting...").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

        def log_message(self, *_):
            pass

    thread = threading.Thread(
        target=lambda: socketserver.TCPServer(("", 7860), Handler).serve_forever(),
        daemon=True,
    )
    thread.start()
    print("Health server started on :7860", flush=True)


def pause_space(token: str) -> None:
    url = f"https://huggingface.co/api/spaces/{SPACE_ID}/pause"
    req = urllib.request.Request(
        url, method="POST",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"Space paused (HTTP {resp.status})", flush=True)
    except Exception as e:
        print(f"Warning: could not self-pause Space: {e}", flush=True)


status = {"message": "Training in progress — check Space logs for step updates."}
start_health_server(status)

if FLAG.exists():
    status["message"] = "Training already complete. Space will pause shortly."
    print("Flag found — training already done. Idling.", flush=True)
    threading.Event().wait()

env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

result = subprocess.run(
    [
        sys.executable, "train.py",
        "--hub-dataset", "cemalec/paraphraser-triples",
        "--epochs", "3",
        "--batch-size", "8",
        "--grad-accum", "4",
        "--max-seq-len", "256",
        "--push-adapter-to", "cemalec/paraphraser-adapter",
        "--resume-from", "cemalec/paraphraser-adapter:checkpoint-1236",
        "--output", "/tmp/adapter",
    ],
    env=env,
    check=False,
)

if result.returncode != 0:
    status["message"] = f"Training FAILED (exit {result.returncode}). Check logs."
    print(status["message"], flush=True)
    sys.exit(result.returncode)

FLAG.touch()
status["message"] = "Training complete! Adapter pushed. Pausing Space now."
print(status["message"], flush=True)

hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    pause_space(hf_token)
else:
    print("HF_TOKEN not set — cannot self-pause. Pause manually.", flush=True)

threading.Event().wait()
