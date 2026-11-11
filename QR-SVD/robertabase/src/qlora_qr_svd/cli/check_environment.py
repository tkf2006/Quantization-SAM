from __future__ import annotations

import argparse
import json

from qlora_qr_svd.environment import collect_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the remote Linux training environment")
    parser.add_argument("--no-strict", action="store_true", help="Print issues without failing")
    args = parser.parse_args()
    try:
        report = collect_environment(strict=not args.no_strict)
    except RuntimeError:
        report = collect_environment(strict=False)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

