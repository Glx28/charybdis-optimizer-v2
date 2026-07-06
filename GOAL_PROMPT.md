You are improving the Charybdis v2 split keyboard layout to be the best possible human-focused layout for daily use. Working directory: /home/nos/charybdis/charybdis-optimizer-v2. Use .venv/bin/python3.

EACH TURN, do these steps:

**STEP 1 — Check optimizer**
`ps aux | grep run_evolution | grep -v grep`
If dead: `python3 tools/best.py` → copy best genome to build/v2_local_search_result.json as {"genome":[...],"total_score":null,"source":"recovered_gapX"} → restart:
`nohup .venv/bin/python3 run_evolution.py > /tmp/run_v20.log 2>&1 &`

**STEP 2 — Progress**
`grep "global best improved" /tmp/run_v20.log | tail -5 && grep "Gen [0-9]" /tmp/run_v20.log | tail -1`
If no gap improvement in 5000+ gens: kill, write best as warmstart, restart.

**STEP 3 — Export for coach**
```
python3 /home/nos/charybdis/charybdis-tools/runtime/evolved_v2_export/promote.py --apply
```
Then in ZMK Studio: run apply_every_key.js → verify_every_key.js → `promote.py --mark-verified --commit --push`
Coach visualizes the layout at http://127.0.0.1:8765/ — check it for layer coherence and gap issues.

**STEP 4 — Analysis (build tools, don't run them each turn — just check output)**
```
python3 tools/human_audit.py    # 7-section GOOD/WARN/BAD report
python3 tools/check.py          # gap, G[], mouse hops, thumb cluster
```
Prefer writing/extending tools in tools/ over inline evaluation — these get reused every turn.

**STEP 5 — Think like a real user. Ask:**
- Hold left thumb 1 hop = instant mouse mode? MB1, MB2, scroll-hold at eff=0 on L10?
- Momentary scroll is REQUIRED on mouse layer — any `@scroll:LX:hold` on a right-hand non-thumb position on L10 works. The kernel enforces this via natural_mouse_layer_exists. The export maps all scroll shortcuts to `&mo 11` (L11 firmware scroll layer).
- Ctrl+V, Ctrl+C, Shift+Enter at eff=0.0 (home row)? These are used 65, 52, 99 times/day.
- Do Ctrl+C and Ctrl+V sit near each other on L1? (Ctrl+C→Ctrl+V is the #1 bigram sequence)
- VSCode shortcuts (Ctrl+Shift+P, F5, F12) — do they cluster on one layer?
- Nav keys (Ctrl+Left/Right, Home, End, PgUp/PgDn) — together on one layer?
- Every toggle-accessed layer has a return toggle (@L0:return)?

**STEP 6 — Fix BAD items**
Rules:
- Only adjust weights or shortcut_importance_overrides in config_v2.yaml
- Never hardcode positions or manually set genome values
- Test each change: reload evaluator, eval best genome, confirm G=[0,0,0,0,0] and gap shift reasonable
- Build token-efficient scripts in tools/ when repeated checks would waste tokens
- To apply: write best genome to warmstart → delete build/v2_scale_factors.json → restart

**STEP 7 — Log**
Print: gen, gap, G=[...], what changed, what to watch next turn.

---

KEY FACTS:
- gap = total_fitness + 49.30; more negative = better. Scale factors recalibrate on restart — cross-run gap comparisons not valid after config changes.
- Current run: gap=-5.068 at gen~19000 (best at gen4379), stagnant 14621 gens. Killed and restarted with new fitness changes (2026-07-05).
- Mouse layer = L10, 1 hold-hop from L0 via left thumb; MB1-5 on R-finger only (not R-thumb)
- Mouse layer FIXED POSITIONS (kernel x-preference): MB1 → x=8 (index), MB2 → x=9 (middle), scroll → x=10 (ring). Penalty: 12000 per column off. At x=11 instead of x=8: 36000 penalty (near-mandatory constraint while still allowing evolution).
- Scroll priority on L10: MB1 > MB2 > momentary scroll > MB4 > MB5
- Scroll enforcement: any `@scroll:LX:hold` placed on L10 (right-hand, non-thumb) satisfies natural_mouse_layer_exists hard constraint. Kernel penalizes scroll effort with coeff 15000 — scroll at eff=1.75 costs 26250 (worse than no scroll at 25000), forcing optimizer to find eff=0 slot.
- SCROLL FIRMWARE ARCHITECTURE (2026-07-05): L11 is the dedicated transparent scroll mode layer. `scroll-layers = <11>` in charybdis_right.overlay. All `@scroll:LX:hold` shortcuts export as `&mo 11` in export_and_analyze_linux.py. L11 is all-transparent so current layer bindings stay visible while scrolling. Firmware is already pushed to GitHub — needs flash after next CI build.
- Mouse layer without scroll = NOT a valid mouse layer (natural_mouse_layer_exists hard constraint, weight=200000)
- L7 frozen (arrows/numpad) — never touch
- Hard constraints: missing_important, layer7_access, natural_mouse_layer_exists, layer_reachability, toggle_back_to_l0
- G must always stay [0,0,0,0,0]
- Real usage (per day): Space=1998, Backspace=592, Ctrl+Backspace=197 (transparent combo), Enter=107, Shift+Enter=99, Ctrl+V=65, Ctrl+C=52, Ctrl+A=7, Ctrl+S=4
- Top bigrams: Ctrl+C→Ctrl+V (27x), Shift+Enter→Ctrl+V (15x), Ctrl+A→Ctrl+C (5x)
- Importance overrides: Ctrl+S=14.0, Ctrl+Y=12.0, Shift+Enter=9.5, Ctrl+A=6.0 (no @scroll hardcode — kernel handles scroll dynamically)
- Scroll effort coefficient: 15000 (raised from 8000 this session)
- Coach app: http://127.0.0.1:8765/ reads charybdis-coach/data/keybindings_explained.csv
- Kernel changes applied: scroll effort coeff 400→8000→15000, empty position penalty restored (L1-L10 non-L7), scroll effort min-tracking bug fixed, LeftAlt/arrow key loader bug fixed, MB1/MB2/scroll x-position preference (12000 per column, 2026-07-05)

---

STOP when ALL true and confirmed in transcript:
1. python3 tools/human_audit.py — zero BAD items
2. python3 tools/best.py — gap ≤ -10.0
3. G=[0,0,0,0,0]
4. Ctrl+V, Ctrl+C, Shift+Enter all at eff=0.0
5. Any `@scroll:LX:hold` on mouse layer at eff=0.0
6. Export ran and coach shows coherent layer clusters
OR: 20 turns elapsed — report final state and stop.
