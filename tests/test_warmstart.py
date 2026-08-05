"""Warmstart genome loading: validation against the current data snapshot."""
import json

from run_evolution import load_warmstart_genome


def _write(tmp_path, genome):
    p = tmp_path / "ws.json"
    p.write_text(json.dumps({"genome": list(genome)}))
    return str(p)


def test_missing_file_returns_none(tmp_path):
    g, ws = load_warmstart_genome(str(tmp_path / "nope.json"), 4, 3)
    assert g is None and ws is None


def test_valid_genome_passthrough(tmp_path):
    g, _ = load_warmstart_genome(_write(tmp_path, [0, -1, 2, 1]), 4, 3)
    assert g.tolist() == [0, -1, 2, 1]


def test_out_of_range_sids_masked_to_empty(tmp_path):
    # Regression 2026-08-05: a stale warmstart artifact written against an
    # older data snapshot carried sids >= n_shortcuts (293..295 with 293
    # shortcuts). Loaded unvalidated, they crashed surrogate training with a
    # CUDA embedding index assertion. Invalid entries must become empty (-1).
    g, _ = load_warmstart_genome(_write(tmp_path, [0, 3, 5, -1]), 4, 3)
    assert g.tolist() == [0, -1, -1, -1]


def test_length_mismatch_rejected(tmp_path):
    g, ws = load_warmstart_genome(_write(tmp_path, [0, 1, 2]), 4, 3)
    assert g is None and ws is None
