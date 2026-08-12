"""Generate optimizer and coach shortcut JSONs from canonical sources."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from shortcut_corpus import (
    DEFAULT_COACH_PATH,
    DEFAULT_OPTIMIZER_PATH,
    DEFAULT_SOURCES_DIR,
    generate_coach_json,
    generate_optimizer_json,
    load_all_apps,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shortcut corpus outputs.")
    parser.add_argument(
        "--sources",
        type=Path,
        default=DEFAULT_SOURCES_DIR,
        help="Directory containing canonical per-app source JSONs.",
    )
    parser.add_argument(
        "--optimizer-out",
        type=Path,
        default=DEFAULT_OPTIMIZER_PATH,
        help="Path to write app_shortcut_scores.json.",
    )
    parser.add_argument(
        "--coach-out",
        type=Path,
        default=DEFAULT_COACH_PATH,
        help="Path to write app_shortcut_reference.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary but do not write files.",
    )
    args = parser.parse_args()

    apps = load_all_apps(args.sources)

    optimizer_data = generate_optimizer_json(apps)
    optimizer_data["timestamp"] = datetime.now(timezone.utc).isoformat()

    coach_data = generate_coach_json(apps)

    if args.dry_run:
        print(f"Would generate {optimizer_data['summary']}")
        print(f"Would write {args.optimizer_out} and {args.coach_out}")
        return

    args.optimizer_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.optimizer_out, "w", encoding="utf-8") as f:
        json.dump(optimizer_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    args.coach_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.coach_out, "w", encoding="utf-8") as f:
        json.dump(coach_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = optimizer_data["summary"]
    print(
        f"Generated {summary['total_apps']} apps, "
        f"{summary['total_shortcuts']} shortcuts "
        f"({summary['total_mapped']} mapped, {summary['total_unmapped']} unmapped)."
    )
    print(f"  Optimizer: {args.optimizer_out}")
    print(f"  Coach:     {args.coach_out}")


if __name__ == "__main__":
    main()
