"""
Space entry point.

Starts an HTTP server on :7860 immediately (satisfies HF's 30-min health check),
then runs training in the main thread. A flag file prevents re-training if the
Space restarts after a successful run.
"""
import http.server
import os
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

FLAG = Path("/tmp/.training_done")


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


status = {"message": "Training in progress — check Space logs for step updates."}
start_health_server(status)

if FLAG.exists():
    status["message"] = "Training already complete. Pause this Space to stop billing."
    print("Flag found — training already done. Idling.", flush=True)
    threading.Event().wait()  # block forever

env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

result = subprocess.run(
    [
        sys.executable, "train.py",
        "--hub-dataset", "cemalec/paraphraser-triples",
        "--epochs", "1",
        "--batch-size", "8",
        "--grad-accum", "4",
        "--max-seq-len", "256",
        "--push-adapter-to", "cemalec/paraphraser-adapter",
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
status["message"] = "Training complete! Adapter pushed to cemalec/paraphraser-adapter. Pause this Space to stop billing."
print(status["message"], flush=True)
threading.Event().wait()  # block forever, keep health server alive
