"""
vision_bridge.py — transport shim: expose the vision pipeline's latest vision_v1 JSON over HTTP
so the browser emotion-display can consume it.

ADDITIVE BY DESIGN. This file does NOT modify any existing vision code or the vision_v1 output.
In real mode it only READS `main.LATEST_VISION_JSON` (the pipeline already refreshes it every
frame) and re-serves it; the perception logic is untouched.

Modes:
    python vision_bridge.py            # run the REAL pipeline (needs a camera) and serve on :8765
    python vision_bridge.py --mock     # no camera: serve a synthetic vision_v1 stream (dev/testing)

Endpoints (CORS: * — localhost dev):
    GET /vision   -> latest vision_v1 JSON (or "{}" before the first frame)
    GET /health   -> {"ok": true, "mode": "real"|"mock"}

The web app polls /vision at ~10 Hz; see sakhi-emotion-display/src/react/useVisionBridge.ts.
"""

import argparse
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8765

# Shared, single-writer state. `json` is the latest vision_v1 string; `mode` labels the source.
_latest = {"json": None, "mode": "real"}


def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")

        def log_message(self, *_args):  # keep the console quiet
            pass

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/health"):
                body = json.dumps({"ok": True, "mode": _latest["mode"]})
            elif self.path.startswith("/vision"):
                body = _latest["json"] or "{}"
            else:
                self.send_response(404)
                self._cors()
                self.end_headers()
                return
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _serve(port):
    httpd = ThreadingHTTPServer(("0.0.0.0", port), _make_handler())
    print(f"[bridge] serving vision_v1 on http://localhost:{port}/vision  (mode={_latest['mode']})",
          file=sys.stderr)
    httpd.serve_forever()


# ── real mode: re-publish what the pipeline already computes ────────────────────────────────────
def run_real(args):
    import main as vision_main  # importing is safe: main.py guards its loop behind __main__

    _latest["mode"] = "real"

    def pump():
        while True:
            _latest["json"] = vision_main.LATEST_VISION_JSON
            time.sleep(0.05)

    threading.Thread(target=pump, daemon=True).start()
    threading.Thread(target=_serve, args=(args.port,), daemon=True).start()

    # Build exactly the args the pipeline's run() expects; force headless JSON output.
    ns = argparse.Namespace(
        camera=args.camera, width=640, height=480, alpha=0.4, fps=30.0,
        num_faces=1, obj_rate=2.0, force_rate=0.0, output="json",
        baseline_file=None, include_baseline=False, session_id="",
        no_display=True, fps_overlay=False, profile=False,
    )
    vision_main.run(ns)  # blocks on the main thread until Ctrl-C / camera close


# ── mock mode: synthesize a vision_v1 stream (no camera) ─────────────────────────────────────────
def run_mock(args):
    # use the REAL formatter so the mock doc matches the production schema exactly
    from outputs.vision_output import format_vision_v1, new_session_id

    _latest["mode"] = "mock"
    threading.Thread(target=_serve, args=(args.port,), daemon=True).start()

    sid = new_session_id()
    print("[bridge] MOCK mode — synthesizing vision_v1 (no camera). Ctrl-C to stop.", file=sys.stderr)
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            # a calm, present guest: mild positive mood that drifts SLOWLY, not a mood swing.
            # (the real camera feed is what makes the face expressive; this is just a steady idle.)
            valence = round(0.18 + 0.10 * math.sin(t * 0.12), 3)   # ~0.08..0.28, very slow
            arousal = round(0.40 + 0.07 * math.sin(t * 0.10), 3)   # ~0.33..0.47, very slow
            ec = (t % 14.0) < 2.0                                  # a brief glance every ~14s
            state = {
                "face_present": True,
                "gru_valence": valence,
                "gru_arousal": arousal,
                "gru_dominance": 0.5,
                "speaking": (t % 18.0) < 3.0,                      # an occasional short utterance
                "eye_contact": ec,
                "gaze_x": round(0.12 * math.sin(t * 0.2), 3),      # eyes mostly settled
                "gaze_y": 0.0,
                "gaze_direction": "center" if ec else "left",
                "eye_contact_score": 0.85 if ec else 0.2,
            }
            _latest["json"] = format_vision_v1(
                state, session_id=sid, emit_reason="mock", timestamp_ms=time.time() * 1000.0,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def main():
    p = argparse.ArgumentParser(description="vision_v1 → HTTP bridge for the emotion-display web app")
    p.add_argument("--mock", action="store_true", help="serve a synthetic stream (no camera needed)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--camera", type=int, default=0, help="camera index for real mode")
    args = p.parse_args()
    if args.mock:
        run_mock(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
