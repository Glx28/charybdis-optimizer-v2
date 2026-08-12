"""Validate shortcut corpus coverage and treat large gaps as critical errors."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shortcut_corpus import (
    CRITICAL_GAP_THRESHOLD,
    CRITICAL_RATIO_THRESHOLD,
    DEFAULT_SOURCES_DIR,
    load_all_apps,
    validate_coverage,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate shortcut corpus coverage.")
    parser.add_argument(
        "--sources",
        type=Path,
        default=DEFAULT_SOURCES_DIR,
        help="Directory containing canonical per-app source JSONs.",
    )
    parser.add_argument(
        "--critical-gap",
        type=int,
        default=CRITICAL_GAP_THRESHOLD,
        help="Gap larger than this is critical.",
    )
    parser.add_argument(
        "--critical-ratio",
        type=float,
        default=CRITICAL_RATIO_THRESHOLD,
        help="Coverage ratio below this is critical.",
    )
    args = parser.parse_args()

    apps = load_all_apps(args.sources)
    gaps = validate_coverage(apps, args.critical_gap, args.critical_ratio)

    if not gaps:
        print("No apps have expected_count set; nothing to validate.")
        print("Set expected_count in each source file and re-run.")
        sys.exit(0)

    missing_expected = [a.name for a in apps if a.expected_count <= 0]
    critical = [g for g in gaps if g.critical]
    ok = [g for g in gaps if not g.critical]

    print("Coverage validation")
    print("=" * 60)
    if missing_expected:
        print(f"\nApps without expected_count ({len(missing_expected)}):")
        for name in missing_expected:
            app = next(a for a in apps if a.name == name)
            print(f"  {name}: {len(app.shortcuts)} shortcuts, expected_count=0")

    if critical:
        print(f"\nCRITICAL GAPS ({len(critical)}):")
        for g in critical:
            print(
                f"  {g.app}: {g.actual}/{g.expected} "
                f"({g.ratio:.0%}) — missing {g.gap} shortcuts"
            )
    else:
        print("\nNo critical gaps.")

    if ok:
        print(f"\nApps within tolerance ({len(ok)}):")
        for g in ok:
            print(
                f"  {g.app}: {g.actual}/{g.expected} "
                f"({g.ratio:.0%}) — missing {g.gap}"
            )

    if critical or missing_expected:
        print("\nFAILED: critical gaps or missing expected_count.")
        sys.exit(1)
    print("\nPASSED.")


if __name__ == "__main__":
    main()
