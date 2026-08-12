"""Shared shortcut corpus utilities for charybdis-optimizer-v2."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TOOLS_DIR = Path(__file__).parent.resolve()
OPTIMIZER_ROOT = TOOLS_DIR.parent
DEFAULT_SOURCES_DIR = OPTIMIZER_ROOT / "data" / "shortcuts_source"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Shortcut:
    keys: str
    action: str
    category: str = "general"
    frequency: str = "medium"
    importance: float = 5.0
    preferred_hand: str = "either"
    mapped: bool = False
    match_count: int = 0
    best_match: dict | None = None
    access_score: int = 0
    weighted_access: int = 0
    freq_weight: int = 5

    def to_optimizer_dict(self) -> dict[str, Any]:
        return {
            "keys": self.keys,
            "action": self.action,
            "category": self.category,
            "frequency": self.frequency,
            "freq_weight": self.freq_weight,
            "importance": self.importance,
            "mapped": self.mapped,
            "match_count": self.match_count,
            "best_match": self.best_match,
            "access_score": self.access_score,
            "weighted_access": self.weighted_access,
        }

    def to_coach_dict(self) -> tuple[str, str]:
        return (self.keys, self.action)


@dataclass
class App:
    id: str
    name: str
    exe_names: list[str] = field(default_factory=list)
    expected_count: int = 0
    shortcuts: list[Shortcut] = field(default_factory=list)
    user_weight: int = 1

    def to_optimizer_dict(self) -> dict[str, Any]:
        mapped = [s for s in self.shortcuts if s.mapped]
        unmapped = [s for s in self.shortcuts if not s.mapped]
        return {
            "id": self.id,
            "name": self.name,
            "user_weight": self.user_weight,
            "total_shortcuts": len(self.shortcuts),
            "mapped": len(mapped),
            "unmapped": len(unmapped),
            "coverage_pct": round(len(mapped) / len(self.shortcuts) * 100) if self.shortcuts else 0,
            "weighted_score": sum(s.importance * s.freq_weight for s in self.shortcuts if s.mapped),
            "weighted_max": sum(s.importance * s.freq_weight for s in self.shortcuts),
            "efficiency_pct": 0,
            "unmapped_high_frequency": [],
            "shortcuts": [s.to_optimizer_dict() for s in self.shortcuts],
        }

    def to_source_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "exe_names": self.exe_names,
            "expected_count": self.expected_count or len(self.shortcuts),
            "shortcuts": [
                {
                    "keys": s.keys,
                    "action": s.action,
                    "category": s.category,
                    "frequency": s.frequency,
                    "importance": s.importance,
                    "preferred_hand": s.preferred_hand,
                }
                for s in self.shortcuts
            ],
        }

    @classmethod
    def from_source_dict(cls, data: dict[str, Any]) -> App:
        shortcuts = []
        for sc in data.get("shortcuts", []):
            shortcuts.append(
                Shortcut(
                    keys=sc["keys"],
                    action=sc["action"],
                    category=sc.get("category", "general"),
                    frequency=sc.get("frequency", "medium"),
                    importance=float(sc.get("importance", 5.0)),
                    preferred_hand=sc.get("preferred_hand", "either"),
                )
            )
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            exe_names=data.get("exe_names", []),
            expected_count=data.get("expected_count", 0),
            shortcuts=shortcuts,
            user_weight=data.get("user_weight", 1),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slugify(text: str) -> str:
    """Create a filesystem-safe slug from app name/id."""
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    return s.strip("_")


def _source_path(app_id: str, sources_dir: Path) -> Path:
    return sources_dir / f"{app_id}.json"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_all_apps(sources_dir: Path | None = None) -> list[App]:
    sources_dir = sources_dir or DEFAULT_SOURCES_DIR
    if not sources_dir.exists():
        return []
    apps = []
    for path in sorted(sources_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        apps.append(App.from_source_dict(data))
    return apps


def save_app(app: App, sources_dir: Path | None = None) -> None:
    sources_dir = sources_dir or DEFAULT_SOURCES_DIR
    sources_dir.mkdir(parents=True, exist_ok=True)
    path = _source_path(app.id, sources_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(app.to_source_dict(), f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def generate_optimizer_json(apps: list[App]) -> dict[str, Any]:
    from datetime import datetime, timezone

    app_dicts = [a.to_optimizer_dict() for a in apps]
    total_shortcuts = sum(len(a.shortcuts) for a in apps)
    total_mapped = sum(1 for a in apps for s in a.shortcuts if s.mapped)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_apps": len(apps),
            "total_shortcuts": total_shortcuts,
            "total_mapped": total_mapped,
            "total_unmapped": total_shortcuts - total_mapped,
            "overall_coverage_pct": round(total_mapped / total_shortcuts * 100) if total_shortcuts else 0,
        },
        "apps": app_dicts,
    }


def generate_coach_json(apps: list[App]) -> dict[str, Any]:
    result: dict[str, Any] = {"_comment": "Per-app keyboard shortcut reference, independent of the optimized layout. Generated from charybdis-optimizer-v2/data/shortcuts_source/."}
    for app in apps:
        shortcuts: dict[str, str] = {}
        for sc in app.shortcuts:
            key = sc.keys
            # Simple dedup: if key already exists, append action
            if key in shortcuts:
                shortcuts[key] += f" (+1 other app-specific meanings)"
            else:
                shortcuts[key] = sc.action
        result[app.name] = {
            "exeNames": app.exe_names,
            "shortcuts": shortcuts,
        }
    return {"apps": result}
