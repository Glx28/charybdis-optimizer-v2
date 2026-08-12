import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\nos\charybdis-optimizer-v2\tools")
import shortcut_corpus as sc

src = Path(r"C:\Users\nos\charybdis-optimizer-v2\data\shortcuts_source")
src.mkdir(exist_ok=True)

data = json.load(open(r"C:\Users\nos\charybdis-optimizer-v2\data\app_shortcut_scores.json", encoding="utf-8"))
apps = data.get("apps", [])
print(f"Bootstrapping {len(apps)} apps...")

total = 0
for a in apps:
    app_id = sc.slugify(a.get("id") or a["name"])
    shortcuts = []
    for s in a.get("shortcuts", []):
        shortcuts.append(
            sc.Shortcut(
                keys=s["keys"],
                action=s["action"],
                category=s.get("category", "general"),
                frequency=s.get("frequency", "medium"),
                importance=s.get("importance", 5.0),
                preferred_hand=s.get("preferred_hand", "either"),
            )
        )
    total += len(shortcuts)
    app = sc.App(
        id=app_id,
        name=a["name"],
        exe_names=a.get("exe_names", []),
        expected_count=a.get("total_shortcuts", 0),
        shortcuts=shortcuts,
        user_weight=a.get("user_weight", 1),
    )
    sc.save_app(app, src)
    print(f"  {app_id}: {len(shortcuts)} shortcuts")

print(f"Done. Total shortcuts bootstrapped: {total}")
