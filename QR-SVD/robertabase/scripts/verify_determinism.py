from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IGNORED_KEYS = {"elapsed_seconds"}


def normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalized(item)
            for key, item in value.items()
            if key not in IGNORED_KEYS
        }
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_hash(run_dir: Path, checkpoint: str) -> str:
    # Archived records retain hashes, not checkpoint tensors.
    saved_hashes = run_dir / "checkpoint_hashes.json"
    if not (run_dir / checkpoint).exists() and saved_hashes.is_file():
        return str(read_json(saved_hashes)[checkpoint]["sha256"])
    candidates = sorted((run_dir / checkpoint).glob("*.safetensors"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one safetensors adapter in {run_dir / checkpoint}, got {candidates}"
        )
    return sha256(candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Require two deterministic repeats to match exactly"
    )
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checks = {
        "summary_without_wall_time": normalized(read_json(args.left / "summary.json"))
        == normalized(read_json(args.right / "summary.json")),
        "metrics_without_wall_time": normalized(read_jsonl(args.left / "metrics.jsonl"))
        == normalized(read_jsonl(args.right / "metrics.jsonl")),
        "best_adapter_sha256": adapter_hash(args.left, "best_adapter")
        == adapter_hash(args.right, "best_adapter"),
        "last_adapter_sha256": adapter_hash(args.left, "last_adapter")
        == adapter_hash(args.right, "last_adapter"),
    }
    payload = {
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "ignored_fields": sorted(IGNORED_KEYS),
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit("Determinism verification failed; rho sweep was not started")


if __name__ == "__main__":
    main()
