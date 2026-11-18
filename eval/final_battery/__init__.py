"""Final-model realism battery: run once per completely trained model (``python -m eval.cli final_battery``).

See README.md in this folder. Kits under ``kits/`` are the 17 Nov 2026 reference implementations; ``runner`` drives
them, ``summary`` writes the one-page read with the GL512 450k reference column, ``acceptance`` compares a run on
GLB450e2_m with the committed reference tables.
"""
