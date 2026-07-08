#!/usr/bin/env python3
import json, re, sys

data = json.load(sys.stdin)
cmd = ""
tool_input = data.get("tool_input") or {}
if isinstance(tool_input, dict):
    cmd = tool_input.get("command", "") or ""

bad = [
    r"\brm\s+-rf\s+/",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bshutdown\b",
    r"\breboot\b"
]

if any(re.search(p, cmd) for p in bad):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "Destructive command blocked by policy."
        }
    }))
else:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}))
