"""``eval.cli verdict``: the gate code of the trusted evaluation suite (docs note
``epics/misc/20261119_trusted_evaluation_suite.md``, s4; build step B1 of s8).

Readers write ``gates_<P|A|B|C>.csv`` from existing tables (no new diagnostic); ``merge`` applies s4 and prints one
``VERDICT: PASS | FAIL | INCONCLUSIVE | INVALID`` line. Modules: ``stats`` (lead bands, date blocks, jackknife, t
table, states), ``gates`` (ids, margins, csv), ``preflight`` (P1-P5), ``reader_a`` / ``reader_b`` / ``reader_c``,
``merge``.
"""
