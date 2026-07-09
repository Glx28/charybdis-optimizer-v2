"""Group shape constants for arrow and Norwegian extra-key cluster atomicity.

Offsets define the expected relative position of each key within its cluster,
keyed by the order/type identifier used in fitness/kernel.py.

Norwegian extra keys: the 5 physical keys that differ between Norwegian and
US International keyboard layouts.  Converted to their US International HID
names and placed at their Norwegian physical positions.
Order 1-5 relative to order-2 (EQUALS AND PLUS) as anchor.

Arrow cluster: types 1-4 (LEFT/RIGHT/UP/DOWN) relative to LEFT as anchor.
"""

# Norwegian extra-key cluster offsets: order → (dx, dy) relative to EQUALS anchor.
# Norwegian physical positions converted to US International key names:
#   §/| → Grave Accent and Tilde    at Norwegian (0,1)
#   +/? → Dash and Underscore       at Norwegian (1,1)
#   ´/` → Equals and Plus           at Norwegian (2,1)  ← anchor
#   ¨/^ → Right Brace               at Norwegian (0,2)
#   </> → Backslash and Pipe        at Norwegian (0,4)
NORWEGIAN_CLUSTER_OFFSETS = {
    1: (-1, 0),   # DASH AND UNDERSCORE
    2: (0, 0),    # EQUALS AND PLUS  (anchor)
    3: (-2, 0),   # GRAVE ACCENT AND TILDE
    4: (-2, 1),   # RIGHT BRACE
    5: (-2, 3),   # BACKSLASH AND PIPE
}

# Visual layout (relative to EQUALS anchor at col 0, row 0):
#   dx: -2   -1    0
# dy=0: [Grv] [Dsh] [Eql]
# dy=1: [RBr]
# dy=3: [Bsl]

# Base key names (uppercase) mapped to their cluster order — mirrors kernel.py
NORWEGIAN_BASE_KEY_ORDER = {
    "DASH AND UNDERSCORE": 1,
    "EQUALS AND PLUS": 2,
    "GRAVE ACCENT AND TILDE": 3,
    "RIGHT BRACE": 4,
    "BACKSLASH AND PIPE": 5,
}

# Arrow cluster: two valid shapes, both relative to LEFT arrow as anchor (0,0).
# Shape A: all on same row  →  L . U . D . R
# Shape B: T-cluster        →  L D R (row 0) + U above D (row -1)
ARROW_SHAPES = [
    # shape A: same-row  (L, D, U, R left-to-right; kernel checks L<R in x)
    {1: (0, 0), 4: (1, 0), 3: (2, 0), 2: (3, 0)},   # LEFT=1, RIGHT=2, UP=3, DOWN=4
    # shape B: T-cluster (L at 0, U one above D, D right of L, R right of D)
    {1: (0, 0), 4: (1, 0), 3: (1, -1), 2: (2, 0)},
]

ARROW_BASE_KEY_TYPE = {
    "LEFTARROW": 1,
    "RIGHTARROW": 2,
    "UPARROW": 3,
    "DOWNARROW": 4,
}


def is_valid_arrow_cluster(positions_by_type: dict, pos_layer: list, pos_x: list, pos_y: list) -> bool:
    """Check if assigned arrow positions form a valid cluster.

    Args:
        positions_by_type: {arrow_type: [pos_idx, ...]} for types present in genome.
        pos_layer, pos_x, pos_y: per-position arrays from layout.

    Returns True if exactly 4 types present, all on same layer, in a valid shape.
    """
    if len(positions_by_type) != 4:
        return False
    layers = set()
    rep = {}  # type → (x, y) using first assigned position
    for atype, idxs in positions_by_type.items():
        for idx in idxs:
            layers.add(pos_layer[idx])
            if atype not in rep:
                rep[atype] = (pos_x[idx], pos_y[idx])
    if len(layers) != 1:
        return False
    # Check each valid shape
    left_xy = rep.get(1)  # LEFT is anchor
    if left_xy is None:
        return False
    lx, ly = left_xy
    for shape in ARROW_SHAPES:
        match = True
        for atype, (dx, dy) in shape.items():
            expected = (lx + dx, ly + dy)
            actual = rep.get(atype)
            if actual is None or abs(actual[0] - expected[0]) > 0.5 or abs(actual[1] - expected[1]) > 0.5:
                match = False
                break
        if match:
            return True
    return False


def is_valid_completion_cluster(positions_by_order: dict, pos_layer: list, pos_x=None, pos_y=None) -> bool:
    """Check the fixed 5-key Norwegian extra-key cluster.

    The cluster may move to any mutable non-L7 layer and any valid anchor, but
    its relative shape is fixed by NORWEGIAN_CLUSTER_OFFSETS.
    """
    if len(positions_by_order) != 5:
        return False
    layers = set()
    reps = {}
    for order, idxs in positions_by_order.items():
        for idx in idxs:
            layers.add(pos_layer[idx])
            reps.setdefault(order, idx)
    if len(layers) != 1:
        return False
    if pos_x is None or pos_y is None:
        return True
    anchor_idx = reps.get(2)
    if anchor_idx is None:
        return False
    ax = pos_x[anchor_idx]
    ay = pos_y[anchor_idx]
    for order, (dx, dy) in NORWEGIAN_CLUSTER_OFFSETS.items():
        idx = reps.get(order)
        if idx is None:
            return False
        if abs(pos_x[idx] - (ax + dx)) > 0.5 or abs(pos_y[idx] - (ay + dy)) > 0.5:
            return False
    return True
