"""Exercise the R3 SDK in Chromium at a real prefixed loopback route."""

import argparse
import base64
import json
from pathlib import Path
from threading import Thread
from time import perf_counter

from ship_perception.manual_review.frame_session import FrameSession
from ship_perception.manual_review.server import LocalServer
from ship_perception.manual_review.source_identity import identify
from ship_perception.manual_review.workbench_auto import AutomaticRun
from ship_perception.manual_review.workbench_server import STATIC
from ship_perception.manual_review.workbench_v3 import WorkbenchV3
from tools import mhr_browser_check as driver
from tools.mhr_synthetic import rect, save_pcd, synthesized

BASE = "/review/hatch"


def read(browser, field):
    return browser.js(f"document.querySelector('[data-id={field}]')?.textContent")


def click(browser, field):
    browser.js(
        f"document.querySelector('[data-id={field}]').scrollIntoView({{block:'nearest'}})"
    )
    browser.click(f"[data-id={field}]")


def screen(browser, frame, xy):
    p = frame.projection
    return browser.js(
        """(() => {
        const r=document.querySelector('[data-id=viewport]').getBoundingClientRect();
        const dims=%s, s=Math.min(r.width/dims[0], r.height/dims[1]);
        return %s.map(([x,y])=>[r.x+(r.width-dims[0]*s)/2+x*s,r.y+(r.height-dims[1]*s)/2+y*s]);
        })()"""
        % (json.dumps([p.nx, p.ny]), json.dumps(p.uv(xy).tolist()))
    )


def screenshot(browser, evidence, name):
    data = browser.call("Page.captureScreenshot", format="png")["data"]
    (evidence / f"{name}.png").write_bytes(base64.b64decode(data))


def exercise(browser, server, evidence):
    review = server.review
    native = review.auto.snapshot["status"] == "READY"
    browser.call(
        "Emulation.setDeviceMetricsOverride",
        width=1400,
        height=950,
        deviceScaleFactor=1.25,
        mobile=False,
    )
    browser.call("Page.navigate", url=server.origin + BASE + "/example#" + server.token)
    browser.wait(
        "document.querySelector('[data-id=source]')?.textContent.startsWith('构建 ')"
    )
    time_started = perf_counter()
    if native:
        key, target = next(iter(review.auto.snapshot["targets"].items()))
        expected = target["geometry"]
        xy = expected["xy"]
        import numpy as np

        center = np.mean(xy, axis=0)
        where = screen(browser, server.frame, [center])[0]
        browser.mouse("mousePressed", *where, button="left", clickCount=1)
        browser.mouse("mouseReleased", *where, button="left", clickCount=1)
        before = read(browser, "coordinates")
        click(browser, "confirm")
        browser.wait(
            "document.querySelector('[data-id=selectionStatus]')?.textContent.includes('已确认自动框')"
        )
        assert before == read(browser, "coordinates")
        assert review.view()["heads"][key]["geometry"] == expected
        assert review.view()["heads"][key]["state"] == "ACCEPTED_AUTO"
        screenshot(browser, evidence, "01-auto-accepted")
        click(browser, "edit")
        viewport = screen(browser, server.frame, [xy[0], xy[2]])
        browser.drag(*viewport)
        browser.wait(
            """(() => {
             const s=document.querySelector('[data-id=status]').textContent;
             return s && !s.includes('正在') && !s.includes('拖动') && s;
        })()"""
        )
        failure = read(browser, "status")
        assert review.view()["heads"][key]["geometry"] == expected
        screenshot(browser, evidence, "02-real-attempt")
        click(browser, "cancel")
        kind = "REAL_AUTO_AND_REJECT"
    else:
        import numpy as np

        theta = 0.16
        truth = rect(np.array([12.0, -22.0]), theta, 18.0, 9.0)
        # Set angle from the actual two displayed points, then draw in that basis.
        click(browser, "add")
        click(browser, "orient")
        ends = screen(browser, server.frame, truth[:2])
        for point in ends:
            browser.mouse("mousePressed", *point, button="left", clickCount=1)
            browser.mouse("mouseReleased", *point, button="left", clickCount=1)
        rough = truth + np.array([0.12, -0.06])
        browser.drag(*screen(browser, server.frame, rough[[0, 2]]))
        browser.wait(
            "document.querySelector('[data-id=selectionStatus]')?.textContent.includes('局部结果待确认')"
        )
        before = read(browser, "coordinates")
        assert browser.js(
            "document.querySelector('polygon[data-label=重测候选]')!==null"
        )
        screenshot(browser, evidence, "01-synthetic-guided")
        click(browser, "confirm")
        browser.wait(
            "document.querySelector('[data-id=selectionStatus]')?.textContent.includes('已确认局部重测')"
        )
        (item,) = review.view()["heads"].values()
        # Screen UV reverses physical XY winding. Compare against the known
        # screen-draft order while retaining the CREATE anchor in the revision.
        truth_in_screen_order = truth[[3, 2, 1, 0]]
        error = max(
            np.linalg.norm(
                np.array(item["geometry"]["xy"]) - truth_in_screen_order, axis=1
            )
        )
        assert item["geometry"]["corner_mapping"]["request_vertices"] == [0, 1, 2, 3]
        assert error <= 0.05
        failure = None
        kind = "SYNTHETIC_RAW_CREATE"
    browser.call("Page.reload")
    browser.wait(
        "document.querySelector('[data-id=selectionStatus]')?.textContent.includes('已确认')"
    )
    assert read(browser, "coordinates") == before
    screenshot(browser, evidence, "03-after-refresh")
    return dict(
        kind=kind,
        browser=browser.call("Browser.getVersion")["product"],
        dpr=browser.js("devicePixelRatio"),
        base_path=BASE,
        real_reject=failure,
        elapsed_s=perf_counter() - time_started,
        reload_equal=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("chromium", type=Path)
    p.add_argument("evidence", type=Path)
    p.add_argument("--pcd", type=Path)
    p.add_argument("--import-run", type=Path)
    args = p.parse_args()
    if bool(args.pcd) != bool(args.import_run):
        p.error(
            "Supply both --pcd and --import-run, or neither for the synthetic fixture."
        )
    args.evidence.mkdir(parents=True, exist_ok=True)
    args.review = False
    pcd = args.pcd
    if pcd is None:
        import numpy as np

        pcd = args.evidence / "synthetic.pcd"
        save_pcd(pcd, synthesized(rect(np.array([12.0, -22.0]), 0.16, 18.0, 9.0)))
    frame = FrameSession(pcd)
    auto = AutomaticRun()
    if args.import_run:
        auto.import_result(args.import_run, frame)
        assert auto.snapshot["status"] == "READY"
    driver.exercise = exercise
    with LocalServer(
        frame,
        review=WorkbenchV3(frame, args.evidence, auto, identify()),
        static_assets=STATIC,
        base_path=BASE,
    ) as server:
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            r = driver.run(args, server)
        finally:
            server.shutdown()
    (args.evidence / "browser.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(r)


if __name__ == "__main__":
    main()
