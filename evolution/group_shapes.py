"""Group shape constants for the Norwegian extra-key cluster.

Offsets define the expected relative position of each key within its cluster,
keyed by the order/type identifier used in fitness/kernel.py.

Norwegian extra keys: the 5 physical keys that differ between Norwegian and
US International keyboard layouts.  Converted to their US International HID
names and placed at their Norwegian physical positions.
Order 1-5 relative to order-2 (EQUALS AND PLUS) as anchor.

"""

# Norwegian raw-key cluster offsets: order → (dx, dy) relative to EQUALS anchor.
# Norwegian physical positions converted to US International key names:
#   §/| → Grave Accent and Tilde    at Norwegian (0,1)
#   +/? → Dash and Underscore       at Norwegian (1,1)
#   ´/` → Equals and Plus           at Norwegian (2,1)  ← anchor
#   ¨/^ → Right Brace               at Norwegian (0,2)
#   </> → Non-US Backslash and Pipe at Norwegian (0,4)
NORWEGIAN_CLUSTER_OFFSETS = {
    1: (-1, 0),   # DASH AND UNDERSCORE
    2: (0, 0),    # EQUALS AND PLUS  (anchor)
    3: (-2, 0),   # GRAVE ACCENT AND TILDE
    4: (-2, 1),   # RIGHT BRACE
    5: (-2, 3),   # NON-US BACKSLASH AND PIPE
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
    "NON-US BACKSLASH AND PIPE": 5,
}

def is_valid_completion_cluster(positions_by_order: dict, pos_layer: list, pos_x=None, pos_y=None) -> bool:
    """Check the fixed 5-key Norwegian raw-key cluster.

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
