from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate best scores from experiment outputs")
    parser.add_argument("root", nargs="?", default="outputs")
    parser.add_argument("--output", default="outputs/aggregate.json")
    args = parser.parse_args()

    records = []
    for path in sorted(Path(args.root).glob("*/summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        record["path"] = str(path.parent)
        records.append(record)

    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(record["method"], []).append(float(record["best_score"]))
    summary = {
        method: {
            "runs": len(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "scores": scores,
        }
        for method, scores in grouped.items()
    }
    payload = {"groups": summary, "runs": records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
