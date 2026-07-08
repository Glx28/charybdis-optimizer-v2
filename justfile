

ai-status:
    @bash .agent-tools/ai-status.sh

ai-context:
    @bash .agent-tools/repo-map.sh

ai-guard:
    @bash .agent-tools/ai-guard.sh

ai-smoke:
    @bash .agent-tools/ai-smoke.sh

ai-tools:
    @bash .agent-tools/ai-tools.sh

ai-research:
    @cat .ai/context/agent-tooling-research.md 2>/dev/null || echo "Run research first."

ai-install-check:
    @kimi --version 2>&1; claude --version 2>&1; codex --version 2>&1; rg --version 2>&1 | head -1; fd --version 2>&1 | head -1; sg --version 2>&1 | head -1; just --version 2>&1 | head -1; repomix --version 2>&1 | head -1; uv --version 2>&1 | head -1; ruff --version 2>&1 | head -1; pytest --version 2>&1 | head -1; pre-commit --version 2>&1 | head -1; biome --version 2>&1 | head -1; tokei --version 2>&1 | head -1; node --version 2>&1; npm --version 2>&1; python3 --version 2>&1

# Generate full run report bundle under build/run_report/
run-report:
    @.venv/bin/python tools/generate_run_report.py

# Audit latest or specified checkpoint
checkpoint-audit CKPT="latest":
    @.venv/bin/python tools/checkpoint_audit.py {{CKPT}}

# Layer profile for latest or specified checkpoint
layer-profile CKPT="latest":
    @.venv/bin/python tools/layer_profile.py {{CKPT}}

# Shortcut audit for latest or specified checkpoint
shortcut-audit CKPT="latest":
    @.venv/bin/python tools/shortcut_audit.py {{CKPT}}

# Hard-constraint trace across checkpoints
constraint-trace:
    @.venv/bin/python tools/constraint_trace.py

# Generate minimal context pack for Kimi sessions
quick-context:
    @bash .agent-tools/quick-context.sh
