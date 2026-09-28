"""Append the stage 1 log entry to the design note and update its Next Step line.

Only those two edits; every other line of the note is left exactly as it was.
"""
from pathlib import Path

NOTE = Path("/home/ecm5702/dev/docs/epics/downscaling-aifs-crps/in-progress/"
            "20260909_station_head_adapter_design.md")
ENTRY = Path("/home/ecm5702/agent-work/20260909-station-head-adapter/notes/_logentry.md")

text = NOTE.read_text()
old_next = ("- Next Step: see the Log at the bottom; stage 1 is being built by a "
            "delegated agent and reports there.")
new_next = ("- Next Step: stage 1 is complete (2026-09-09); the manifest, the static "
            "station table, the five station tables and the coverage report are under "
            "`/home/ecm5702/agent-work/20260909-station-head-adapter/outputs/` and "
            "`notes/`. Stage 2a is next: build the 2026 input bundles and run the frozen "
            "branch U checkpoint at step 100,000 over a first slice of the calendar.")
assert text.count(old_next) == 1, "the Next Step line is not what was expected"
text = text.replace(old_next, new_next)

entry = ENTRY.read_text().rstrip() + "\n"
assert text.rstrip().endswith("its report will be appended here."), "unexpected end of note"
text = text.rstrip() + "\n" + entry
NOTE.write_text(text)
print("note updated")
