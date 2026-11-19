"""Drive the genuine loopback component with Chromium pointer input."""

import argparse
import base64
import json
from pathlib import Path
from threading import Thread
from time import monotonic, sleep

from ship_perception.manual_review.frame_session import FrameSession
from ship_perception.manual_review.server import LocalServer, source_sha
from ship_perception.manual_review.workbench import Workbench
from ship_perception.manual_review.workbench_auto import AutomaticRun
from ship_perception.manual_review.workbench_server import STATIC
from tools import mhr_browser_check as driver


def select(key):
    return '[data-id="' + key + '"]'


def exercise(browser, server, out):
    browser.call("Page.navigate", url=server.origin + "/example#" + server.token)
    browser.call(
        "Emulation.setDeviceMetricsOverride",
        width=1280,
        height=850,
        deviceScaleFactor=1.25,
        mobile=False,
    )
    auto = server.review.auto
    if auto.snapshot["status"] != "DISABLED":
        deadline = monotonic() + 320
        while monotonic() < deadline:
            if auto.snapshot["status"] in ("READY", "FAILED"):
                break
            sleep(0.75)
        assert auto.snapshot["status"] == "READY", auto.snapshot.get("reason")
        browser.wait(
            "document.querySelector('[data-id=coordinates]')?.children.length===4"
        )
        initial = browser.js(
            "document.querySelector('[data-id=coordinates]').innerText"
        )
        picked = browser.js("document.querySelector('[data-id=target]').value")
        browser.click(select("confirm"))
        browser.wait(
            "document.querySelector('[data-id=selectionStatus]').textContent.includes('已确认自动框')"
        )
        assert (
            browser.js("document.querySelector('[data-id=coordinates]').innerText")
            == initial
        )
        browser.call("Page.reload")
        browser.wait(
            "document.querySelector('[data-id=selectionStatus]')?.textContent.includes('已确认自动框')"
        )
        assert server.review.view()["heads"][picked]["state"] == "ACCEPTED_AUTO"
        result = dict(path="native_auto_accept", target_id=picked, reloaded=True)
    else:
        browser.wait(
            "document.querySelector('[data-id=auto]')?.textContent.includes('未启用')"
        )
        browser.click(select("add"))
        p = server.frame.projection
        rect = p.uv([[3.24, -17.72], [20.92, -26.42]]).tolist()
        coords = browser.js(
            """(() => {
         const r=document.querySelector('[data-id=viewport]').getBoundingClientRect();
         const dims=%s, s=Math.min(r.width/dims[0], r.height/dims[1]);
         return %s.map(([x,y])=>[r.x+(r.width-dims[0]*s)/2+x*s,r.y+(r.height-dims[1]*s)/2+y*s]);
        })()"""
            % (json.dumps([p.nx, p.ny]), json.dumps(rect))
        )
        browser.drag(*coords)
        browser.wait(
            "document.querySelector('[data-id=selectionStatus]')?.textContent.includes('局部结果待确认')"
        )
        pending = browser.js(
            "document.querySelector('[data-id=coordinates]').innerText"
        )
        assert browser.js(
            "document.querySelector('polygon[data-label=重测候选]')!==null"
        )
        browser.click(select("confirm"))
        browser.wait(
            "document.querySelector('[data-id=selectionStatus]')?.textContent.includes('已确认局部重测')"
        )
        assert (
            browser.js("document.querySelector('[data-id=coordinates]').innerText")
            == pending
        )
        browser.call("Page.reload")
        browser.wait(
            "document.querySelector('[data-id=selectionStatus]')?.textContent.includes('已确认局部重测')"
        )
        result = dict(
            path="guided_create", reloaded=True, geometry_unchanged_after_confirm=True
        )
    screenshot = browser.call("Page.captureScreenshot", format="png")["data"]
    (out / "workbench.png").write_bytes(base64.b64decode(screenshot))
    result["browser"] = browser.call("Browser.getVersion")["product"]
    result["dpr"] = browser.js("devicePixelRatio")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("chromium", type=Path)
    p.add_argument("pcd", type=Path)
    p.add_argument("evidence", type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--import-run", type=Path)
    group.add_argument("--run-auto", action="store_true")
    args = p.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    args.review = False
    frame = FrameSession(args.pcd)
    auto = AutomaticRun("RUNNING" if args.run_auto else "DISABLED")
    if args.import_run:
        auto.import_result(args.import_run, frame)
    if args.run_auto:
        Thread(
            target=auto.run,
            args=(args.pcd.resolve(), args.evidence, frame),
            daemon=True,
        ).start()
    driver.exercise = exercise
    with LocalServer(
        frame,
        review=Workbench(frame, args.evidence, auto, source_sha()),
        static_assets=STATIC,
    ) as server:
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            result = driver.run(args, server)
        finally:
            server.shutdown()
            auto.close()
    (args.evidence / "browser.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(result)


if __name__ == "__main__":
    main()
