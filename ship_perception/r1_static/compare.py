"""Compare two static runs without treating either result as ground truth."""
import argparse
import csv
import json
from pathlib import Path

from .run import REPO


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metrics(directory):
    result = _read(directory / "result.json")
    profiles = _read(directory / "profile_debug.json")
    all_profiles = [row for group in profiles.values() for row in group]
    residuals = [h["local_deck"]["residual_p95_m"] for h in result["hatches"]
                 if h["local_deck"]["residual_p95_m"] is not None]
    return dict(result=result, count=result["confirmed_hatch_count"],
                edges=sum(len(h["boundaries"]) for h in result["hatches"]),
                deck_residual=sum(residuals) / len(residuals) if residuals else None,
                profile_rate=sum(bool(p["valid"]) for p in all_profiles) / len(all_profiles) if all_profiles else None,
                runtime=result["timing"]["total_ms"])


def compare(first, second, output):
    first, second = Path(first), Path(second)
    names = sorted({p.name for p in first.iterdir() if (p / "result.json").is_file()} |
                   {p.name for p in second.iterdir() if (p / "result.json").is_file()})
    columns = ("场景", "旧舱数", "新舱数", "旧边数量", "新边数量", "旧Local Deck P95米", "新Local Deck P95米",
               "旧Profile有效率", "新Profile有效率", "旧运行毫秒", "新运行毫秒", "配置差异")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name in names:
            if not (first / name / "result.json").exists() or not (second / name / "result.json").exists():
                continue
            old, new = _metrics(first / name), _metrics(second / name)
            old_cfg, new_cfg = _read(first / name / "resolved_config.json"), _read(second / name / "resolved_config.json")
            changed = [section + "." + key for section, fields in old_cfg.items()
                       if isinstance(fields, dict) for key, value in fields.items()
                       if value != new_cfg[section][key]]
            writer.writerow(dict(zip(columns, (name, old["count"], new["count"], old["edges"], new["edges"],
                                                old["deck_residual"], new["deck_residual"],
                                                old["profile_rate"], new["profile_rate"],
                                                old["runtime"], new["runtime"], ",".join(changed)))))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--output-root", type=Path, default=REPO / "Ship-Unloader-Work/r1_static")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.output_root / (args.first + "_vs_" + args.second + ".csv")
    print(compare(args.output_root / args.first, args.output_root / args.second, output))


if __name__ == "__main__":
    main()
