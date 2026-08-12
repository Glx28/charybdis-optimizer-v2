"""Canonical shortcut corpus utilities.

This module is the single source of truth for building the optimizer's
``app_shortcut_scores.json`` and the coach's ``app_shortcut_reference.json``.
Shortcuts are stored as per-app JSON files in
``charybdis-optimizer-v2/data/shortcuts_source/`` and generated from there.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SOURCES_DIR = Path(__file__).parent.parent / "data" / "shortcuts_source"
DEFAULT_OPTIMIZER_PATH = Path(__file__).parent.parent / "data" / "app_shortcut_scores.json"
DEFAULT_COACH_PATH = (
    Path(__file__).parent.parent.parent / "charybdis-coach" / "data" / "app_shortcut_reference.json"
)

# Frequency mapping used by the optimizer kernel.
FREQ_WEIGHTS = {
    "constant": 10,
    "high": 8,
    "medium": 5,
    "low": 2,
    "rare": 1,
}

# Validation thresholds. A gap of more than this many shortcuts from the
# expected count is treated as a critical error.
CRITICAL_GAP_THRESHOLD = 5
CRITICAL_RATIO_THRESHOLD = 0.50


@dataclass
class Shortcut:
    keys: str
    action: str
    category: str = "general"
    frequency: str = "medium"
    importance: float = 5.0
    preferred_hand: str = "either"
    # Optional override for the HID base key; normally inferred from keys.
    base_key: str = ""
    # Optional aliases for the coach search.
    aliases: list[str] = field(default_factory=list)
    # Optional list of additional apps where this same combo has meaning.
    variants: list[dict[str, Any]] = field(default_factory=list)

    def freq_weight(self) -> int:
        return FREQ_WEIGHTS.get(self.frequency, 1)


@dataclass
class App:
    id: str
    name: str
    user_weight: int = 1
    exe_names: list[str] = field(default_factory=list)
    expected_count: int = 0
    shortcuts: list[Shortcut] = field(default_factory=list)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "app"


def load_app(source_path: Path) -> App:
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return App(
        id=data["id"],
        name=data["name"],
        user_weight=data.get("user_weight", 1),
        exe_names=data.get("exe_names", []),
        expected_count=data.get("expected_count", 0),
        shortcuts=[Shortcut(**sc) for sc in data.get("shortcuts", [])],
    )


def load_all_apps(sources_dir: Path = DEFAULT_SOURCES_DIR) -> list[App]:
    apps: list[App] = []
    if not sources_dir.exists():
        return apps
    for path in sorted(sources_dir.glob("*.json")):
        if path.name.lower() == "schema.json":
            continue
        apps.append(load_app(path))
    return apps


def app_to_source_dict(app: App) -> dict[str, Any]:
    return {
        "id": app.id,
        "name": app.name,
        "user_weight": app.user_weight,
        "exe_names": app.exe_names,
        "expected_count": app.expected_count,
        "shortcuts": [
            {
                "keys": sc.keys,
                "action": sc.action,
                "category": sc.category,
                "frequency": sc.frequency,
                "importance": sc.importance,
                "preferred_hand": sc.preferred_hand,
                **({"base_key": sc.base_key} if sc.base_key else {}),
                **({"aliases": sc.aliases} if sc.aliases else {}),
                **({"variants": sc.variants} if sc.variants else {}),
            }
            for sc in app.shortcuts
        ],
    }


def save_app(app: App, sources_dir: Path = DEFAULT_SOURCES_DIR) -> Path:
    sources_dir.mkdir(parents=True, exist_ok=True)
    path = sources_dir / f"{app.id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(app_to_source_dict(app), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def generate_optimizer_json(apps: list[App]) -> dict[str, Any]:
    """Build app_shortcut_scores.json from canonical app sources."""
    total_shortcuts = sum(len(a.shortcuts) for a in apps)
    total_mapped = sum(
        sum(1 for sc in a.shortcuts if sc.importance >= 6.0) for a in apps
    )
    total_unmapped = total_shortcuts - total_mapped
    coverage_pct = round(total_mapped / total_shortcuts * 100) if total_shortcuts else 0

    out_apps: list[dict[str, Any]] = []
    for app in apps:
        weighted_score = 0.0
        weighted_max = 0.0
        mapped_count = 0
        shortcuts_out: list[dict[str, Any]] = []
        for sc in app.shortcuts:
            fw = sc.freq_weight()
            weighted_max += sc.importance * fw
            # In a generated corpus nothing is placed yet, so access_score is 0.
            weighted_score += 0.0
            if sc.importance >= 6.0:
                mapped_count += 1
            entry: dict[str, Any] = {
                "keys": sc.keys,
                "action": sc.action,
                "category": sc.category,
                "frequency": sc.frequency,
                "freq_weight": fw,
                "importance": sc.importance,
                "mapped": sc.importance >= 6.0,
                "match_count": 0,
                "best_match": None,
                "access_score": 0,
                "weighted_access": 0.0,
            }
            if sc.variants:
                entry["merged_variants"] = sc.variants
            shortcuts_out.append(entry)

        app_total = len(app.shortcuts)
        app_unmapped = app_total - mapped_count
        app_coverage = round(mapped_count / app_total * 100) if app_total else 0
        app_efficiency = round(weighted_score / weighted_max * 100) if weighted_max else 0

        out_apps.append({
            "id": app.id,
            "name": app.name,
            "user_weight": app.user_weight,
            "total_shortcuts": app_total,
            "mapped": mapped_count,
            "unmapped": app_unmapped,
            "coverage_pct": app_coverage,
            "weighted_score": round(weighted_score, 1),
            "weighted_max": round(weighted_max, 1),
            "efficiency_pct": app_efficiency,
            "unmapped_high_frequency": [],
            "shortcuts": shortcuts_out,
        })

    return {
        "timestamp": "",  # Caller should stamp.
        "summary": {
            "total_apps": len(apps),
            "total_shortcuts": total_shortcuts,
            "total_mapped": total_mapped,
            "total_unmapped": total_unmapped,
            "overall_coverage_pct": coverage_pct,
        },
        "apps": out_apps,
    }


def generate_coach_json(apps: list[App]) -> dict[str, Any]:
    """Build app_shortcut_reference.json from canonical app sources."""
    ref: dict[str, Any] = {
        "_comment": (
            "Per-app keyboard shortcut reference, independent of the optimized layout. "
            "Generated from charybdis-optimizer-v2/data/shortcuts_source/."
        ),
        "apps": {},
    }
    for app in apps:
        ref["apps"][app.name] = {
            "exeNames": app.exe_names,
            "shortcuts": {sc.keys: sc.action for sc in app.shortcuts},
        }
    return ref


@dataclass
class CoverageGap:
    app: str
    actual: int
    expected: int
    gap: int
    ratio: float
    critical: bool


def validate_coverage(
    apps: list[App],
    critical_gap: int = CRITICAL_GAP_THRESHOLD,
    critical_ratio: float = CRITICAL_RATIO_THRESHOLD,
) -> list[CoverageGap]:
    gaps: list[CoverageGap] = []
    for app in apps:
        if app.expected_count <= 0:
            continue
        actual = len(app.shortcuts)
        gap = app.expected_count - actual
        ratio = actual / app.expected_count
        critical = gap > critical_gap or ratio < critical_ratio
        gaps.append(CoverageGap(
            app=app.name,
            actual=actual,
            expected=app.expected_count,
            gap=gap,
            ratio=ratio,
            critical=critical,
        ))
    return sorted(gaps, key=lambda g: g.ratio)
