"""Loopback-only offline UI. Request payloads and private paths are never logged."""

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from secrets import token_urlsafe
import re
from threading import BoundedSemaphore

from .contract import FLAGS, ContractError, canonical, parse_json, selection
from .frame_session import FrameSession

MAX_JSON = 16_384
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 12

    def __init__(self, frame, port=0, review=None, static_assets=None, *, base_path=""):
        if not re.fullmatch(r"(?:/[A-Za-z0-9_-]+)*", base_path):
            raise ContractError("INVALID_BASE_PATH")
        self.base_path = base_path
        self.frame = frame
        self.review = review
        self.token = token_urlsafe(32)
        self.slots = BoundedSemaphore(12)
        ui = Path(__file__).with_name("ui")
        self.static = {
            url: (mime, (ui / file).read_bytes())
            for url, (file, mime) in (static_assets or STATIC).items()
        }
        super().__init__(("127.0.0.1", port), Handler)
        self.host = "127.0.0.1:%d" % self.server_port
        self.origin = "http://" + self.host

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            request.settimeout(4)
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_error(self, code, message=None, explain=None):
        self.json({"error": "HTTP_ERROR"}, code)

    def reply(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        self.wfile.write(data)

    def json(self, payload, status=200):
        self.reply(canonical({**payload, **FLAGS}), "application/json", status)

    def authorize(self, write=False, private=True):
        if self.headers.get_all("Host") != [self.server.host]:
            raise ContractError("BAD_HOST", 403)
        origins = self.headers.get_all("Origin")
        if (write or origins is not None) and origins != [self.server.origin]:
            raise ContractError("BAD_ORIGIN", 403)
        if private:
            values = self.headers.get_all("X-MHR-Token")
            if (
                not values
                or len(values) != 1
                or not hmac.compare_digest(
                    values[0].encode("utf-8"), self.server.token.encode("ascii")
                )
            ):
                raise ContractError("BAD_TOKEN", 403)

    def routed_path(self):
        prefix = self.server.base_path
        if not self.path.startswith(prefix + "/"):
            raise ContractError("ROUTE_NOT_FOUND", 404)
        return self.path[len(prefix):]

    def do_GET(self):
        try:
            path = self.routed_path()
            self.authorize(private=path not in self.server.static)
            frame = self.server.frame
            if path in self.server.static:
                mime, data = self.server.static[path]
                self.reply(data, mime)
            elif path in ("/api/v2/view", "/api/v2/revisions", "/api/v3/view") and hasattr(
                self.server.review, "read"
            ):
                self.json(self.server.review.read(path))
            elif path == "/api/view":
                self.json(
                    {
                        "manifest": frame.manifest,
                        "targets": frame.targets,
                        **(
                            self.server.review.view()
                            if self.server.review is not None
                            else {}
                        ),
                    }
                )
            elif path in {"/layer/" + k + ".png" for k in frame.layers}:
                self.reply(frame.layers[path[7:-4]], "image/png")
            else:
                raise ContractError("ROUTE_NOT_FOUND", 404)
        except ContractError as exc:
            self.json({"error": str(exc)}, exc.status)

    def do_POST(self):
        try:
            self.authorize(write=True)
            if self.headers.get_all("Transfer-Encoding") is not None:
                raise ContractError("UNSUPPORTED_TRANSFER_ENCODING")
            length = self.headers.get_all("Content-Length")
            if (
                not length
                or len(length) != 1
                or not length[0].isascii()
                or not length[0].isdigit()
            ):
                raise ContractError("INVALID_LENGTH")
            if len(length[0]) > 6 or not 0 < int(length[0]) <= MAX_JSON:
                raise ContractError("JSON_SIZE", 413)
            if self.headers.get_content_type() != "application/json":
                raise ContractError("JSON_REQUIRED", 415)
            data = self.rfile.read(int(length[0]))
            if len(data) != int(length[0]):
                raise ContractError("INCOMPLETE_JSON")
            payload = parse_json(data)
            frame = self.server.frame
            path = self.routed_path()
            if path == "/api/selection":
                xy = selection(payload, frame.manifest, frame.projection, frame.targets)
                self.json(
                    {
                        "request_id": payload["request_id"],
                        "xy": xy.tolist(),
                        "view_id": frame.manifest["view_id"],
                        "state": "VALID",
                    }
                )
            elif self.server.review is not None:
                self.json(self.server.review.handle(path, payload))
            else:
                raise ContractError("ROUTE_NOT_FOUND", 404)
        except ContractError as exc:
            self.json({"error": str(exc)}, exc.status)
        except (ValueError, RecursionError):
            self.json({"error": "INVALID_REQUEST"}, 400)
        except OSError:
            self.json({"error": "LOCAL_IO_ERROR"}, 500)


def source_sha():
    from .source_identity import identify

    return identify()["build_sha"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcd", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell-m", type=float, default=0.1)
    parser.add_argument("--frame-id", default="pcd")
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    from .revisions import Review

    try:
        frame = FrameSession(args.pcd, args.cell_m, args.frame_id, args.overlay)
        with LocalServer(
            frame, args.port, Review(frame, args.output, source_sha())
        ) as server:
            print("MHR_URL=" + server.origin + "/#" + server.token, flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (OSError, UnicodeError, KeyError, ValueError) as exc:
        parser.exit(
            2,
            "MHR_START_ERROR="
            + (str(exc) if isinstance(exc, ContractError) else type(exc).__name__)
            + "\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
