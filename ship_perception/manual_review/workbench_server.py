"""Launch the offline MHR workbench for one trusted local PCD."""

import argparse
from pathlib import Path
from threading import Thread

from .contract import ContractError
from .frame_session import FrameSession
from .server import LocalServer, source_sha
from .workbench import Workbench
from .workbench_auto import AutomaticRun

HTML = "text/html; charset=utf-8"
JS = "text/javascript; charset=utf-8"
STATIC = {
    "/": ("workbench.html", HTML),
    "/example": ("example.html", HTML),
    "/workbench.js": ("workbench.js", JS),
    "/bootstrap.js": ("bootstrap.js", JS),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcd", type=Path)
    parser.add_argument("output", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--import-run", type=Path, help="Trusted native result directory"
    )
    group.add_argument(
        "--skip-auto",
        action="store_true",
        help="Open for a local-only CREATE investigation",
    )
    parser.add_argument("--frame-id", default="pcd")
    parser.add_argument("--cell-m", default=0.1, type=float)
    parser.add_argument("--port", default=0, type=int)
    parser.add_argument("--protocol", choices=(2, 3), default=3, type=int)
    parser.add_argument("--base-path", default="")
    args = parser.parse_args()
    auto = AutomaticRun(
        "DISABLED" if args.skip_auto else "LOADING" if args.import_run else "RUNNING"
    )
    try:
        pcd = args.pcd.resolve()
        frame = FrameSession(pcd, args.cell_m, args.frame_id)
        if args.protocol == 3:
            from .source_identity import identify
            from .workbench_v3 import WorkbenchV3

            workbench = WorkbenchV3(frame, args.output, auto, identify())
        else:
            workbench = Workbench(frame, args.output, auto, source_sha() or "UNVERIFIED")
        if args.import_run:
            callback, params = auto.import_result, (args.import_run, frame)
        else:
            callback, params = auto.run, (pcd, args.output, frame)
        if not args.skip_auto:
            Thread(target=callback, args=params, daemon=True).start()
        with LocalServer(frame, args.port, workbench, STATIC, base_path=args.base_path) as server:
            print(f"MHR_R{args.protocol}_URL=" + server.origin + args.base_path + "/#" + server.token, flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(
            2,
            "MHR_R2_START_ERROR="
            + (str(exc) if isinstance(exc, ContractError) else type(exc).__name__)
            + "\n",
        )
    finally:
        auto.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
