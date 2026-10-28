
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


class _SnapshotStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.telemetry: dict[str, Any] = {
            "connection": "webots-http-bridge",
            "simulation": "Starting",
            "joints": [],
        }
        self.jpeg: bytes | None = None
        self.updated_at: float | None = None
        self.frame_at: float | None = None

    def publish(
        self,
        *,
        telemetry: dict[str, Any] | None = None,
        camera_jpeg: bytes | None = None,
    ) -> None:
        now = time.time()
        with self.lock:
            if telemetry is not None:
                # Serialize now to ensure the published object is JSON-safe and
                # detach it from any mutable objects owned by the controller.
                self.telemetry = json.loads(json.dumps(telemetry, allow_nan=False))
                self.updated_at = now
            if camera_jpeg is not None:
                if not isinstance(camera_jpeg, (bytes, bytearray)):
                    raise TypeError("camera_jpeg must be bytes")
                self.jpeg = bytes(camera_jpeg)
                self.frame_at = now

    def get_telemetry(self) -> dict[str, Any]:
        with self.lock:
            result = dict(self.telemetry)
            result["telemetry_timestamp"] = self.updated_at
            result["camera_timestamp"] = self.frame_at
            result["camera_available"] = self.jpeg is not None
            return result

    def get_camera(self) -> bytes | None:
        with self.lock:
            return self.jpeg

    def get_snapshot_metadata(self) -> dict[str, Any]:
        with self.lock:
            return {
                "telemetry": dict(self.telemetry),
                "telemetry_timestamp": self.updated_at,
                "camera_timestamp": self.frame_at,
                "camera_available": self.jpeg is not None,
            }


class WebotsHTTPBridge:
    """Threaded HTTP server plus a thread-safe latest-snapshot store."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = int(port)
        self._store = _SnapshotStore()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def publish(
        self,
        *,
        telemetry: dict[str, Any] | None = None,
        camera_jpeg: bytes | None = None,
    ) -> None:
        """Call from the Webots controller thread after reading device values."""
        self._store.publish(telemetry=telemetry, camera_jpeg=camera_jpeg)

    def start(self) -> None:
        """Start serving in a daemon thread; returns immediately."""
        if self._thread and self._thread.is_alive():
            return

        store = self._store

        class Handler(BaseHTTPRequestHandler):
            server_version = "TIAGoWebotsBridge/1.0"

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = urlparse(self.path).path

                if path == "/health":
                    body = json.dumps({
                        "ok": True,
                        "service": "tiago-webots-http-bridge",
                        "camera_available": store.get_camera() is not None,
                    }).encode("utf-8")
                    self._send(200, body, "application/json; charset=utf-8")
                    return

                if path == "/api/telemetry":
                    body = json.dumps(
                        store.get_telemetry(), allow_nan=False
                    ).encode("utf-8")
                    self._send(200, body, "application/json; charset=utf-8")
                    return

                if path == "/api/snapshot":
                    body = json.dumps(
                        store.get_snapshot_metadata(), allow_nan=False
                    ).encode("utf-8")
                    self._send(200, body, "application/json; charset=utf-8")
                    return

                if path == "/api/camera.jpg":
                    jpeg = store.get_camera()
                    if jpeg is None:
                        self._send(
                            503,
                            b'{"error":"No camera frame has been published yet"}',
                            "application/json; charset=utf-8",
                        )
                    else:
                        self._send(200, jpeg, "image/jpeg")
                    return

                self._send(
                    404,
                    b'{"error":"Not found","endpoints":["/health","/api/telemetry","/api/camera.jpg","/api/snapshot"]}',
                    "application/json; charset=utf-8",
                )

            def log_message(self, format: str, *args: Any) -> None:
                # Avoid flooding the Webots console for every dashboard refresh.
                return

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError as exc:
            raise RuntimeError(
                f"Could not start HTTP bridge at {self.host}:{self.port}. "
                "Check whether the port is already in use."
            ) from exc

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="webots-http-bridge",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Optional clean shutdown."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None


def capture_jpeg(camera: Any, path: str, quality: int = 80) -> bytes | None:
    """
    Save a Webots Camera frame as JPEG and return its bytes.

    This intentionally uses Webots' built-in saveImage() rather than Pillow,
    so the controller doesn't need an extra image-encoding dependency.
    Call it sssssswwwwadafrom the controller thread after a successful robot.step().
    """
    result = camera.saveImage(path, int(quality))
    if result != 0:
        print(f"Camera saveImage failed: return code={result}, path={path}")
        return None

    with open(path, "rb") as image_file:
        return image_file.read()
