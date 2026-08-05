# Codex CLI Entry Point

Read `AGENTS.md` — it is the canonical project-rules file for all agents (GPU policy, dynamic layer assignment, 30k-generation run target, agent tooling rules).

The previous Codex-specific prompt here was removed: it hardcoded a fixed mouse layer (L10), fixed key coordinates, and a stale stop condition, all of which contradict the dynamic-layer rules in `AGENTS.md`.
