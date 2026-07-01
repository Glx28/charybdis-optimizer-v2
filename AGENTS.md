# Charybdis Optimizer V2

Evolutionary layout optimizer for the Charybdis keyboard.

## Dynamic Layer Assignment

Only L0 and L7 have stable semantic roles. L0 is base typing/thumb access. L7 is frozen fallback/game/system-safe space.

Every other layer is dynamically assigned during each generation. Do not encode assumptions such as a fixed mouse layer, scroll layer, travel layer, app layer, code layer, Excel layer, DMS layer, or system layer by layer number.

Mouse, scroll-mode access, app workflows, completion keys, and duplicates must be scored from actual bindings, usage data, access paths, physical effort, and workflow support. Behavior names such as `coach_l2_hold`, `coach_l6_toggle`, `coach_mouse_lock`, and `coach_travel_toggle` are access/beacon names only; they do not prove the target layer's role.
