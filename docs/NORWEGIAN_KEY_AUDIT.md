# Norwegian key-pool audit

Target: Windows Norwegian (`nb-NO`) with the keyboard emitting HID usages. The
host layout turns those usages into glyphs; English-US output is not assumed.

## Findings

- L0 already has direct HID access for `ø/Ø` (`SemiColon and Colon`), `æ/Æ`
  (`Left Apos and Double`), and `å/Å` (`Left Brace`).
- The optimizer's five-key raw group had the wrong fifth HID usage. It used
  `Backslash and Pipe`, which the Norwegian host map emits as `'` and `*` and
  which is already present on L0. The missing angle-bracket usage is
  `Non-US Backslash and Pipe`: `<` unshifted and `>` with Shift. On Norwegian
  Windows, AltGr on that usage also emits `|`.
- The Coach host map mapped `<` to comma and `>` to period. Those produce
  `,`/`;` and `.`/`:`, respectively. It also mapped browser `IntlBackslash`
  to ordinary Backslash. Both mappings now point to the distinct Non-US HID
  usage. Ordinary `Backslash and Pipe` is recorded as `'`/`*`.
- Other Norwegian-specific raw outputs were checked against the same host map:
  `|/§` (`Grave Accent and Tilde`), `+/?` (`Dash and Underscore`),
  `\\`/grave/acute (`Equals and Plus`), and `¨/^/~` (`Right Brace`). These
  remain members of the same five-key raw group in keyboard-like order.
- AltGr symbols on the number row remain accessible through the normal number
  HID usages and Windows Norwegian mapping: `@`, `£`, `$`, `€`, `{`, `[`, `]`,
  and `}`. They are not additional physical key usages for this raw group.
- The optimizer had no dedicated unmodified `LeftAlt` capability. It now
  synthesizes one and acceptance requires it directly on L0, so application
  Alt shortcuts remain available from the base layer.
- Norwegian AltGr output depends on an unmodified RightAlt (AltGr) hold. The
  optimizer now synthesizes this capability and requires it on a layer
  reachable from L0. This preserves number-row third-level outputs such as
  `@`, `£`, `$`, `€`, `{`, `[`, `]`, and `}` without assigning them false
  standalone HID usages.

## Pool and grouping

The raw family remains five keys; its fifth member is corrected to
`Non-US Backslash and Pipe`. The existing exact-shape mutation, compactness
group, scoring metadata, completion audit, and exporter-facing HID name all use
that member. No layer number is assigned to the group: evolution places it on a
reachable non-L0 layer alongside workflow bindings.

## Updated sources

- `core/norwegian_keys.py`, `core/loader.py`
- `evolution/group_shapes.py`, `evolution/completion_cluster.py`,
  `evolution/__init__.py`, `fitness/kernel.py`, `evolution/acceptance.py`
- `data/windows_norwegian_host.json` mirrors in Coach, ZMK config, tools, and
  portable Coach

The next validated run snapshots these files and its progress/audits record
whether the corrected raw-key group, L0 LeftAlt contract, and reachable RightAlt
AltGr contract are satisfied.
