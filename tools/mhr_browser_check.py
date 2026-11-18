"""Exercise the standalone page in Chromium via CDP (websocket-client optional)."""

import argparse
import base64
from contextlib import suppress
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
from urllib.request import urlopen

import websocket

from ship_perception.manual_review.frame_session import FrameSession
from ship_perception.manual_review.server import LocalServer
from ship_perception.manual_review.revisions import Review


class Browser:
    def __init__(self, url):
        self.ws = websocket.create_connection(url, suppress_origin=True, timeout=10)
        self.count = 0

    def call(self, method, **params):
        self.count += 1
        self.ws.send(json.dumps(dict(id=self.count, method=method, params=params)))
        while True:
            result = json.loads(self.ws.recv())
            if result.get("id") == self.count:
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result["result"]

    def js(self, script):
        value = self.call(
            "Runtime.evaluate", expression=script, returnByValue=True, awaitPromise=True
        )
        if "exceptionDetails" in value:
            raise AssertionError(value["exceptionDetails"])
        return value["result"].get("value")

    def wait(self, script):
        deadline = monotonic() + 8
        while monotonic() < deadline:
            result = self.js(script)
            if result:
                return result
            sleep(0.05)
        raise AssertionError(
            (
                script,
                self.js("document.querySelector('#status')?.textContent"),
                self.js("location.origin"),
                self.js("typeof state"),
            )
        )

    def mouse(self, event, x, y, **options):
        self.call("Input.dispatchMouseEvent", type=event, x=x, y=y, **options)

    def click(self, selector):
        x, y = self.js(
            f"(() => {{const r = document.querySelector('{selector}').getBoundingClientRect();"
            "return [r.x + r.width/2, r.y + r.height/2]})()"
        )
        self.mouse("mousePressed", x, y, button="left", clickCount=1)
        self.mouse("mouseReleased", x, y, button="left", clickCount=1)

    def drag(self, a, b):
        self.mouse("mousePressed", *a, button="left", buttons=1, clickCount=1)
        self.mouse("mouseMoved", *b, button="left", buttons=1)
        self.mouse("mouseReleased", *b, button="left", clickCount=1)


def run(args, server):
    with TemporaryDirectory(
        dir=args.evidence.parent, prefix=".mhr-browser-", ignore_cleanup_errors=True
    ) as profile:
        with subprocess.Popen(
            [
                str(args.chromium),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--disable-extensions",
                "--remote-debugging-port=0",
                "--user-data-dir=" + str(Path(profile).resolve()),
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as process:
            try:
                port_file = Path(profile) / "DevToolsActivePort"
                deadline = monotonic() + 12
                while (
                    not port_file.exists()
                    and monotonic() < deadline
                    and process.poll() is None
                ):
                    sleep(0.1)
                port = port_file.read_text().splitlines()[0]
                with urlopen(
                    "http://127.0.0.1:" + port + "/json/list", timeout=5
                ) as url:
                    target = next(
                        t
                        for t in json.load(url)
                        if t["type"] == "page" and t["url"] == "about:blank"
                    )
                browser = Browser(target["webSocketDebuggerUrl"])
                try:
                    return (
                        exercise_review(browser, server, args.evidence)
                        if args.review
                        else exercise(browser, server, args.evidence)
                    )
                finally:
                    with suppress(OSError, websocket.WebSocketException, RuntimeError):
                        browser.call("Browser.close")
                    browser.ws.close()
            finally:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.terminate()


def exercise(browser, server, evidence):
    rows = []
    browser.call("Page.navigate", url=server.origin + "/#" + server.token)
    browser.wait("document.querySelector('#status')?.textContent.includes('拖动绘制')")
    for i, ratio in enumerate((1, 1.25, 2)):
        browser.call(
            "Emulation.setDeviceMetricsOverride",
            width=1163 - i * 61,
            height=711 + i * 29,
            deviceScaleFactor=ratio,
            mobile=False,
        )
        browser.wait(f"devicePixelRatio === {ratio} && !!image")
        browser.click("#reset")
        bounds = browser.js(
            "(() => { const r=viewport.getBoundingClientRect(), c=camera();"
            "return {x:r.x+c.x,y:r.y+c.y,s:c.scale,nx:m.nx,ny:m.ny,"
            "w:r.width,h:r.height} })()"
        )
        x, y, scale, nx, ny = (bounds[k] for k in ("x", "y", "s", "nx", "ny"))
        a, b = [x + 0.28 * nx * scale, y + 0.26 * ny * scale], [
            x + 0.68 * nx * scale,
            y + 0.72 * ny * scale,
        ]
        browser.drag(a, b)
        browser.wait("!document.querySelector('#match').disabled")
        uv = browser.js("vertices()")
        expected = [
            [0.28 * nx, 0.26 * ny],
            [0.68 * nx, 0.26 * ny],
            [0.68 * nx, 0.72 * ny],
            [0.28 * nx, 0.72 * ny],
        ]
        error = max(abs(x - y) for a, b in zip(uv, expected) for x, y in zip(a, b))
        # CDP mouse positions are quantized to CSS pixels by Chromium.
        assert error <= 1.1 / scale, (error, scale)
        browser.mouse("mouseWheel", *a, deltaX=0, deltaY=-122)
        browser.wait("state.zoom > 1")
        browser.click("#pan")
        browser.drag(a, [a[0] + 39, a[1] + 29])
        assert browser.js("JSON.stringify(vertices())") == json.dumps(
            uv, separators=(",", ":")
        )
        browser.js(
            "document.querySelector('#layer').value='density';document.querySelector('#layer').dispatchEvent(new Event('change'));"
        )
        browser.js(
            "document.querySelector('#angle').value='12.5';document.querySelector('#angle').dispatchEvent(new Event('input'));"
        )
        browser.click("#match")
        browser.wait("state.candidate !== null")
        result = browser.js("state.candidate")
        measured = browser.js("vertices().map(world)")
        mapped = result.get("xy", result.get("rough_xy"))
        assert mapped is not None
        cross = max(abs(a - b) for p, q in zip(mapped, measured) for a, b in zip(p, q))
        assert cross <= 1e-8
        stale = browser.js(
            "(async()=>{try {await api('/api/selection',{...state.request,view_id:'stale'});}catch(e){return e.message}})()"
        )
        assert stale == "STALE"
        rows.append(
            dict(
                dpr=ratio,
                mouse_css_error_px=error * scale,
                xy_cross_m=cross,
                result=result.get("state", result.get("match_state")),
                stale=stale,
                resized=bounds,
            )
        )
        if i < 2:
            browser.click("#draw")
            browser.click("#cancel")
    png = browser.call("Page.captureScreenshot", format="png")["data"]
    (evidence / "browser.png").write_bytes(base64.b64decode(png))
    browser.click("#cancel")
    browser.wait("vertices().length === 0")
    return rows


def exercise_review(browser, server, evidence):
    browser.call("Page.navigate", url=server.origin + "/#" + server.token)
    browser.wait("document.querySelector('#status')?.textContent.includes('拖动绘制')")
    browser.call(
        "Emulation.setDeviceMetricsOverride",
        width=1200,
        height=800,
        deviceScaleFactor=1.25,
        mobile=False,
    )
    coords = browser.js(
        """(() => {
        const b=viewport.getBoundingClientRect(), c=camera();
        return [[3.2,-17.7],[20.9,-26.4]].map(x=>project(pixel(x),c))
             .map(([x,y])=>[x+b.left,y+b.top]);
    })()"""
    )
    browser.drag(*coords)
    browser.click("#match")
    browser.wait("state.candidate?.match_state === 'MATCHED_REVIEW'")
    candidate = browser.js("state.candidate")
    png = browser.call("Page.captureScreenshot", format="png")["data"]
    (evidence / "review-candidate.png").write_bytes(base64.b64decode(png))
    browser.click("#accept")
    browser.wait("state.parent === 1 && !!state.saved[state.annotation]")
    annotation = browser.js("state.annotation")
    browser.call(
        "Browser.setDownloadBehavior",
        behavior="allow",
        downloadPath=str(evidence.resolve()),
    )
    browser.click("#export")
    browser.wait("document.querySelector('#status').textContent.includes('已导出')")
    deadline = monotonic() + 5
    while monotonic() < deadline and not list(evidence.glob("*.csv")):
        sleep(0.05)
    assert list(evidence.glob("*.csv")), "CSV file was not downloaded"
    browser.click("#undo")
    browser.wait("state.parent === 2 && state.saved[state.annotation] === null")
    browser.call("Page.reload")
    browser.wait("state.heads && Object.values(state.heads).includes(2)")
    return dict(
        match=candidate["match_state"],
        confirmation=1,
        undo=2,
        reloaded=True,
        annotation=annotation,
        export_csv=True,
        candidate=candidate,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chromium", type=Path)
    parser.add_argument("pcd", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    frame = FrameSession(args.pcd, cell_m=0.1)
    review = Review(frame, args.evidence) if args.review else None
    with LocalServer(frame, review=review) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            rows = run(args, server)
        finally:
            server.shutdown()
    (args.evidence / "browser.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
