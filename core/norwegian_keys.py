"""Norwegian Windows layout metadata for HID/ZMK Studio key parameters.

The optimizer works with physical HID keys. Display characters on the host are
Norwegian OS-layout results and must not be used as Studio parameters.
"""

NORWEGIAN_HID_DISPLAY = {
    "Left Brace": "å",
    "SemiColon and Colon": "ø",
    "Left Apos and Double": "æ",
    "Backslash and Pipe": "\\",
    "Dash and Underscore": "+/?",
    "Equals and Plus": "\\`/´",
    "Grave Accent and Tilde": "|/§",
    "Right Brace": "^/~",
}

LITERAL_TO_HID_PARAMETER = {
    "!": "1 and Bang",
    "@": "2 and At",
    "#": "3 and Hash",
    "$": "4 and Dollar",
    "%": "5 and Percent",
    "^": "6 and Caret",
    "&": "7 and Ampersand",
    "*": "8 and Star",
    "(": "9 and Left Bracket",
    ")": "0 and Right Bracket",
    "-": "Dash and Underscore",
    "_": "Dash and Underscore",
    "=": "Equals and Plus",
    "+": "Equals and Plus",
    "`": "Grave Accent and Tilde",
    "~": "Grave Accent and Tilde",
    "[": "Left Brace",
    "]": "Right Brace",
    "\\": "Backslash and Pipe",
    "|": "Backslash and Pipe",
    ";": "SemiColon and Colon",
    ":": "SemiColon and Colon",
    "'": "Left Apos and Double",
    '"': "Left Apos and Double",
    ",": "Comma and LessThan",
    "<": "Comma and LessThan",
    ".": "Period and GreaterThan",
    ">": "Period and GreaterThan",
    "/": "ForwardSlash and QuestionMark",
    "?": "ForwardSlash and QuestionMark",
}

KEY_TOKEN_ALIASES = {
    "Del": "Delete",
    "Backspace": "Delete",
    "BkSp": "Delete",
    "Enter": "Return Enter",
    "Return": "Return Enter",
    "Space": "Spacebar",
    "Esc": "Escape",
    "Page Up": "PageUp",
    "PageUp": "PageUp",
    "PgUp": "PageUp",
    "Page Down": "PageDown",
    "PageDown": "PageDown",
    "PgDn": "PageDown",
    "Left": "LeftArrow",
    "Right": "RightArrow",
    "Up": "UpArrow",
    "Down": "DownArrow",
}

MULTIWORD_BASE_KEYS = (
    "Page Down",
    "Page Up",
    "Return Enter",
    "Left Brace",
    "Right Brace",
    "Dash and Underscore",
    "Equals and Plus",
    "Grave Accent and Tilde",
    "Backslash and Pipe",
    "SemiColon and Colon",
    "Left Apos and Double",
    "Comma and LessThan",
    "Period and GreaterThan",
    "ForwardSlash and QuestionMark",
)

RAW_COMPLETION_NORWEGIAN = (
    "Dash and Underscore",
    "Equals and Plus",
    "Grave Accent and Tilde",
    "Right Brace",
    "Backslash and Pipe",
)

RAW_COMPLETION_ORDER = {key.upper(): i for i, key in enumerate(RAW_COMPLETION_NORWEGIAN)}


def canonical_hid_parameter(token: str) -> str:
    value = str(token or "").strip()
    if value.startswith("Keyboard "):
        value = value[len("Keyboard "):].strip()
    if value in KEY_TOKEN_ALIASES:
        return KEY_TOKEN_ALIASES[value]
    if value in LITERAL_TO_HID_PARAMETER:
        return LITERAL_TO_HID_PARAMETER[value]
    return value


def parse_shortcut_keys_norwegian(keys: str):
    """Return (modifiers, canonical HID base key) for a shortcut string."""
    text = str(keys or "").strip()
    if not text:
        return [], ""
    if "+" in text:
        if text.endswith("+"):
            return [p for p in text[:-1].split("+") if p], "Equals and Plus"
        parts = [p for p in text.split("+") if p]
        if len(parts) >= 2:
            return parts[:-1], canonical_hid_parameter(parts[-1])
    for base in MULTIWORD_BASE_KEYS:
        suffix = " " + base
        if text == base:
            return [], canonical_hid_parameter(base)
        if text.endswith(suffix):
            return [p for p in text[:-len(suffix)].split() if p], canonical_hid_parameter(base)
    parts = text.split()
    if len(parts) <= 1:
        return [], canonical_hid_parameter(text)
    return parts[:-1], canonical_hid_parameter(parts[-1])

