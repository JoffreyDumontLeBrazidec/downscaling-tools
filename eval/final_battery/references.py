"""GL512 450k EMA-2000 reference values of the final-model battery (row GLB450e2_m, E2 box, 30 dates of 2023,
10 draws, one per input member; 17 Nov 2026).

Sources (docs master, epics/fine-scale-o320-o1280/in-progress/): terrain = 20261117_realism_t2m_terrain.md s3-s6 and
support/20261117_realism_t2m/BATTERY.md; consistency = 20261117_realism_consistency_ensemble_physics.md s2, s5, s6;
wind = 20261117_realism_wind_cyclones.md s2, s3, s6. Text as the summary page prints it.

19 Nov 2026: the acceptance run on Jupiter (battery on GLB450e2_m and GLB450e2_m10) reproduced every committed 17 Nov
table bit for bit (max rel diff 0). Where a number typed in a note differed from what the battery computes from those
same tables (rounding or averaging order: T4, T7, Z4, C1, C9, C12, C13, W3-W5), the computed value is used here, so the
reference column is the battery's own output on GL512 450k. tests/test_final_battery.py recomputes them.
"""

REF = {
    # terrain / surface
    "T1": "15 (sp 15): +13 to +39 Pa above 1500 m and dz > 400 m, +12 to +60 h; no 2t / msl flag",
    "T1b": "+21.1 (29/30)",
    "T1c": "1.12 / 1.17",
    "T2": "0.983 (1,440)",
    "T2b": "-0.19 (92 cases, 22 dates; Norway -0.37)",
    "D2": "0.949 / 0.956",
    "T3": "1.11; 50.1 vs 50.6",
    "T4": "+12 h 0.064, +24 h 0.142; truth 0",  # recomputed from the 17 Nov D tables (note s5 typed 0.140)
    "T5": "< 0.015",
    "T6": "-0.14 (158/180)",
    "T7": "38 / 47; truth 21 -> 61",  # note s3 typed 37 / 46 (other averaging order)
    "Z1a": "1.26 .. 1.01",
    "Z1b": "1.23 .. 1.02",
    "Z1c": "1.10 .. 1.05",
    "Z1d": "3.7-5.9 (46 Pa vs 8-13)",
    "Z3": "117 / 372 / 93",
    "Z4": "1.52 / 1.38",
    "Z6": "1.47 / 1.67 / 1.69",
    "DH": "within 6 %",
    "B11": "< 0.006",
    # ensemble and physical consistency (sd ratios unless said)
    "C1.tp": "1.85 1.62 1.38 1.23 1.18 1.11",
    "C1.msl": "1.23 1.22 1.06 1.08 1.03 1.03",
    "C1.ws": "1.20 1.20 1.07 1.07 1.02 1.02",
    "C1.2t": "1.11 1.16 1.04 1.09 1.02 1.05",
    "C1.tp5": "1.87 1.64 1.37 1.25 1.17 1.09",
    "C2.tp": "1.46 -> 1.18",
    "C2.2t": "1.40 -> 1.11",
    "C2.ws": "1.22 -> 1.05",
    "C2.sea": "1.12 -> 1.01",
    "C3": "0.41",
    "C4": "1.137 / 0.876 / 1.129 / 0.895",
    "C5": "3.30 2.07 1.52 1.28 1.19 1.13 1.05 1.03",
    "C6": "2.5; 35.3 vs 43.7; 9.4 vs 4.1",
    "C7": "1.89",
    "C8": "0.681 vs 0.684; 15.09 vs 15.02; 0.137 vs 0.132",
    "C9": "0.99 / 1.00 / 0.96",
    "C10": "0.484 vs 0.477; 0.258 vs 0.264",
    "C11": "0.97-1.02; 1.25",
    "C12": "0.966",
    "C13": "0.89; 0.944",
    "C14": "not run on 450k (terrain T4: 2d > 2t on 0.06-0.14 % of land)",
    # wind and cyclones
    "W2.24": "0.910 / 0.925 / 0.924 / 0.929",
    "W3.24": "-0.3 / -0.3 / -0.6 / -0.3",
    "W4.24": "1.02 / 0.88 / 0.84 / 0.85",
    "W5": "1.71 vs 1.77 (+24 h), 1.71 vs 1.76 (+72 h)",
    "W7.24": "ws 1.23, msl 1.30",
    # +72 h lines of the wind kit: not quoted in the note, computed from its 17 Nov tables (acceptance run 19 Nov 2026)
    "W2.72": "0.894 / 0.927 / 0.925 / 0.927",
    "W3.72": "-0.2 / -0.3 / -0.3 / -0.4",
    "W4.72": "0.99 / 0.90 / 0.88 / 0.81",
    "W7.72": "0.94-1.03",
    "W8.24": "extratropical 1.55 (0.06, 76 lows); tropical 1.47 (10)",
    "W8.72": "extratropical 1.10; tropical 0.99",
    "W9": ">= 96 kt (500k vs ENS9): log RMW +0.25 (0.05), pmin +4.3 (1.3) hPa",
}
