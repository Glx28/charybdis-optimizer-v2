---
name: testing-verification
description: Use before declaring a task complete or when tests are mentioned.
---

Run the smallest relevant verification command before finishing. For Python: `pytest -q` or `pytest tests/test_v2.py -q` or `ruff check . --output-format=concise`. For CUDA work: run GPU-specific tests if present, then `just ai-guard`. Do not silence real errors by weakening checks.
