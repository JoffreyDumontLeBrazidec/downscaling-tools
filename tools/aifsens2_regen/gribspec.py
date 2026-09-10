"""Field specification for the AIFS ENS version 2 initial conditions and outputs.

This module is the single place where the composition of an initial-condition
file is written down.  Every other module imports the lists from here so that
the retrieval, the assembly and the validation can never drift apart.

An initial condition for one ensemble member and one start consists of two
input times, the analysis six hours before the start (t-6) and the analysis at
the start itself (t0).  Each input time carries 111 fields, so a complete
member file holds 222 fields.  The 111 fields of one input time break down as
follows.

    surface (stream enfo, type pf)                      13 fields
    pressure levels (stream enfo, type pf)   5 x 14  =  70 fields
    specific humidity (stream enfo, type pf) 1 x 13  =  13 fields
    waves (stream waef, type pf)                        11 fields
    constants (stream oper, type an)                     4 fields
                                                       ---
                                                       111 fields

The constants are taken from the operational analysis rather than from the
ensemble because they are invariant fields (orography, land-sea mask and the
two sub-grid orography parameters) that carry no ensemble member number.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# MARS parameter identifiers, grouped exactly as the retrieval requests them.
# --------------------------------------------------------------------------

# Surface parameters: 10u 10v 2d 2t msl tcc skt sp sd tp fscov ci tcw and so on.
PARAM_SFC = [165, 166, 168, 167, 151, 141, 235, 134, 139, 170, 39, 40, 136]

# The four invariant fields: lsm, sdor, slor, z.
PARAM_CON = [172, 160, 163, 129]

# Pressure-level parameters: t u v w z.
PARAM_PL = [130, 131, 132, 135, 129]

# The fourteen pressure levels the checkpoint expects for those five parameters.
LEVELS_PL = [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Specific humidity is carried on thirteen levels only: the 10 hPa level is absent.
PARAM_Q = [133]
LEVELS_Q = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Wave parameters from the perturbed wave ensemble.
PARAM_WAVE = [
    140233, 140114, 140115, 140116, 140117, 140118,
    140119, 140230, 140232, 140229, 140219,
]

# The ten perturbed members.  Member 0, the control, is deliberately excluded:
# the campaign initialises member m from member m's own perturbed analysis and
# never falls back to the control.
MEMBERS = list(range(1, 11))

# The retrieval grid.
GRID = "N320"
AREA = "90/0.0/-90/360"

# --------------------------------------------------------------------------
# Field counts derived from the lists above.
# --------------------------------------------------------------------------

N_SFC_PER_MEMBER_PER_TIME = len(PARAM_SFC)                       # 13
N_PL_PER_MEMBER_PER_TIME = len(PARAM_PL) * len(LEVELS_PL)        # 70
N_Q_PER_MEMBER_PER_TIME = len(PARAM_Q) * len(LEVELS_Q)           # 13
N_WAVE_PER_MEMBER_PER_TIME = len(PARAM_WAVE)                     # 11
N_CON_PER_TIME = len(PARAM_CON)                                  # 4

# The perturbed part of one input time for one member.
N_PERTURBED_PER_MEMBER_PER_TIME = (
    N_SFC_PER_MEMBER_PER_TIME
    + N_PL_PER_MEMBER_PER_TIME
    + N_Q_PER_MEMBER_PER_TIME
    + N_WAVE_PER_MEMBER_PER_TIME
)                                                                # 107

FIELDS_PER_INPUT_TIME = N_PERTURBED_PER_MEMBER_PER_TIME + N_CON_PER_TIME   # 111
FIELDS_PER_MEMBER_FILE = 2 * FIELDS_PER_INPUT_TIME                          # 222

# --------------------------------------------------------------------------
# Retrieval groups.  A group is one MARS request shape.  "per_date_per_time" is
# the number of fields the group returns for one date and one time of day, all
# ten members together, which is what the retrieval uses to compute the
# expected count of a grouped file.
# --------------------------------------------------------------------------

GROUPS = {
    "sfc": dict(
        stream="enfo", type="pf", levtype="sfc",
        param=PARAM_SFC, levelist=None, perturbed=True,
        per_date_per_time=len(MEMBERS) * N_SFC_PER_MEMBER_PER_TIME,        # 130
    ),
    "pl": dict(
        stream="enfo", type="pf", levtype="pl",
        param=PARAM_PL, levelist=LEVELS_PL, perturbed=True,
        per_date_per_time=len(MEMBERS) * N_PL_PER_MEMBER_PER_TIME,         # 700
    ),
    "q": dict(
        stream="enfo", type="pf", levtype="pl",
        param=PARAM_Q, levelist=LEVELS_Q, perturbed=True,
        per_date_per_time=len(MEMBERS) * N_Q_PER_MEMBER_PER_TIME,          # 130
    ),
    "wave": dict(
        stream="waef", type="pf", levtype="sfc",
        param=PARAM_WAVE, levelist=None, perturbed=True,
        per_date_per_time=len(MEMBERS) * N_WAVE_PER_MEMBER_PER_TIME,       # 110
    ),
    "con": dict(
        stream="oper", type="an", levtype="sfc",
        param=PARAM_CON, levelist=None, perturbed=False,
        per_date_per_time=N_CON_PER_TIME,                                   # 4
    ),
}

# The atmospheric groups are mostly online in the archive for January to May
# 2026, while the wave group is largely on tape.  The retrieval submits them as
# two separate SLURM arrays so that an atmospheric request never queues behind
# a tape mount.
ATMOSPHERIC_GROUPS = ["sfc", "pl", "q", "con"]
WAVE_GROUPS = ["wave"]

# --------------------------------------------------------------------------
# The 68 variables the downstream dataset build reads, on the O320 grid.
# --------------------------------------------------------------------------

LANE_PARAM_SFC = [165, 166, 168, 167, 136, 134, 235, 151]        # 8 surface
LANE_PARAM_PL = [133, 130, 131, 132, 135, 129]                   # 6 upper air
LANE_LEVELS_PL = [50, 100, 200, 300, 400, 500, 700, 850, 925, 1000]

LANE_FIELDS_PER_MEMBER_PER_STEP = (
    len(LANE_PARAM_SFC) + len(LANE_PARAM_PL) * len(LANE_LEVELS_PL)
)                                                                # 68

# The forecast produces two lead times.
LEAD_STEPS = [6, 12]
LEAD_TIME_HOURS = 12

LANE_FIELDS_PER_START = (
    LANE_FIELDS_PER_MEMBER_PER_STEP * len(LEAD_STEPS) * len(MEMBERS)
)                                                                # 1360

# The native output of the checkpoint.  This is the value observed on the
# existing complete run at
# /home/ecm5702/scratch/eval/aifs_v1_vs_regen_20260909/regen/v2_2026/full_d20260605_m1.grib
# and it is re-derived from the checkpoint's own typed variables at run time,
# so that a checkpoint change is noticed rather than silently accepted.
NATIVE_FIELDS_PER_STEP_EXPECTED = 119
NATIVE_FIELDS_PER_MEMBER_EXPECTED = NATIVE_FIELDS_PER_STEP_EXPECTED * len(LEAD_STEPS)

# Output GRIB identification for the regenerated ensemble.
OUTPUT_CLASS = "ai"
OUTPUT_STREAM = "enfo"
OUTPUT_TYPE = "pf"
OUTPUT_EXPVER = "rgn2"
OUTPUT_GENERATING_PROCESS = 2

# The model name.  This is NOT written into the GRIB headers, because the
# eccodes installed here, version 2.47.0, has no "model" key: grib_set rejects
# both "model" and "modelName" with "Key/value not found", while class, expver,
# stream, type, number and generatingProcessIdentifier all set correctly.  The
# name is kept here, and recorded in the manifests, so that the intent is not
# lost; if a later eccodes gains the key, add it to the encoding dictionary in
# run_forecasts.base_config and the outputs will carry it.
OUTPUT_MODEL = "aifs-ens"
OUTPUT_MODEL_IS_ENCODED = False
