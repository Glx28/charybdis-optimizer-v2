#!/usr/bin/env python3
"""PreToolUse hook: block destructive Bash commands."""
import json
import re
import sys

FORBIDDEN_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\b.*of=/dev/"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
]


def main():
    payload = json.load(sys.stdin)
    tool_name = payload.get("tool", "")
    arguments = payload.get("arguments", {})
    if tool_name != "Bash":
        json.dump({"allow": True}, sys.stdout)
        return
    command = arguments.get("command", "")
    if any(p.search(command) for p in FORBIDDEN_PATTERNS):
        json.dump(
            {"allow": False, "reason": "Destructive command blocked by project hook."},
            sys.stdout,
        )
        return
    json.dump({"allow": True}, sys.stdout)


if __name__ == "__main__":
    main()
