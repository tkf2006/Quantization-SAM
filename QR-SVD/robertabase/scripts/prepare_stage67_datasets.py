from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "nyu-mll/glue"
DATASET_REVISION = "bcdcba7"
SUPPORTED_TASKS = ("rte", "mrpc", "cola")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and export the pinned GLUE datasets used by Stage 6-7"
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--tasks", nargs="+", choices=SUPPORTED_TASKS, default=SUPPORTED_TASKS)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        target = args.output_root / f"glue_{task}_{DATASET_REVISION}"
        if (target / "dataset_dict.json").is_file():
            print(f"Skip prepared dataset: {target}")
            continue
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"Refusing to overwrite incomplete non-empty directory: {target}")
        dataset = load_dataset(DATASET_NAME, task, revision=DATASET_REVISION)
        dataset.save_to_disk(str(target))
        print(
            f"Saved {task}: train={len(dataset['train'])}, "
            f"validation={len(dataset['validation'])} -> {target}"
        )


if __name__ == "__main__":
    main()
