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


# --------------------------------------------------------------------------
# The broad 108-variable selection.
#
# The summer half of the campaign is retrieved straight from the MARS archive
# with a much wider variable list than the 68 above: fifteen instantaneous
# surface fields, six accumulations, four soil fields on two layers, and the
# five upper-air fields on fourteen pressure levels with specific humidity on
# thirteen.  The early half of the campaign does not exist in the archive and
# has to be built from the regenerated native forecasts instead.  The lists
# below describe that same broad selection as it appears in the native output,
# together with the header changes needed to make the regenerated files carry
# the parameter identifiers that the archive uses.
#
# Nothing here touches the 68-variable lists above: the regeneration still
# reads those, and the two selections live side by side.
# --------------------------------------------------------------------------

# Fifteen instantaneous surface fields.  The native file puts 10u, 10v, 2t, 2d
# on typeOfLevel surface rather than heightAboveGround, and tcc on
# entireAtmosphere, but the shortName is the same either way and the dataset
# build names its variables from the shortName, so no header change is needed.
BROAD_PARAM_SFC_INSTANT = [
    165, 166, 167, 168, 134, 151, 235, 136, 164,
    186, 187, 188, 260289, 228246, 228247,
]

# Six accumulated fields, as the native forecast writes them.
BROAD_PARAM_SFC_ACCUM = [228, 143, 144, 169, 175, 205]

# Four soil fields.  The native file writes them as swvl1, swvl2, stl1 and stl2
# on typeOfLevel depthBelowLandLayer, levels 0 and 7.
BROAD_PARAM_SOIL = [39, 40, 139, 170]

# Upper air.  Geopotential, temperature, the two horizontal wind components and
# the vertical velocity live on all fourteen levels; specific humidity is absent
# at 10 hPa and therefore lives on thirteen.
BROAD_PARAM_PL = [129, 130, 131, 132, 135]
BROAD_LEVELS_PL = [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
BROAD_PARAM_Q = [133]
BROAD_LEVELS_Q = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

BROAD_FIELDS_PER_MEMBER_PER_STEP = (
    len(BROAD_PARAM_SFC_INSTANT)
    + len(BROAD_PARAM_SFC_ACCUM)
    + len(BROAD_PARAM_SOIL)
    + len(BROAD_PARAM_PL) * len(BROAD_LEVELS_PL)
    + len(BROAD_PARAM_Q) * len(BROAD_LEVELS_Q)
)                                                                # 108

BROAD_FIELDS_PER_START = (
    BROAD_FIELDS_PER_MEMBER_PER_STEP * len(LEAD_STEPS) * len(MEMBERS)
)                                                                # 2160

# How a native soil field has to be re-encoded so that it matches the archive.
# The archive holds volumetric soil water as paramId 260199, shortName vsw, and
# soil temperature as paramId 260360, shortName sot, both on typeOfLevel
# soilLayer with the layer number as the level.  The value is
# (new paramId, new level).  The re-encoding also has to raise the message from
# GRIB edition 1 to edition 2, because neither the parameter nor the level type
# exists in edition 1.
BROAD_SOIL_ENCODING = {
    39: (260199, 1),
    40: (260199, 2),
    139: (260360, 1),
    170: (260360, 2),
}

# How a native accumulated field has to be renamed so that it matches the
# archive.  Solar and thermal radiation already agree, so they are absent here.
BROAD_ACCUM_PARAM_RENAME = {
    228: 228228,
    143: 228143,
    144: 228144,
    205: 231002,
}

# The 108 variable names the finished store holds, in the order the store puts
# them, which is plain alphabetical order of the remapped param_level name.
# This list is compared against the store metadata as a final check, and it is
# also what the validation of an O320 file reconstructs from the GRIB headers.
BROAD_STORE_VARIABLES = [
    "100u", "100v", "10u", "10v", "2d", "2t", "cp", "fscov", "hcc", "lcc",
    "mcc", "msl", "q_100", "q_1000", "q_150", "q_200", "q_250", "q_300",
    "q_400", "q_50", "q_500", "q_600", "q_700", "q_850", "q_925", "rowe",
    "sf", "skt", "sot_1", "sot_2", "sp", "ssrd", "strd", "t_10", "t_100",
    "t_1000", "t_150", "t_200", "t_250", "t_300", "t_400", "t_50", "t_500",
    "t_600", "t_700", "t_850", "t_925", "tcc", "tcw", "tp", "u_10", "u_100",
    "u_1000", "u_150", "u_200", "u_250", "u_300", "u_400", "u_50", "u_500",
    "u_600", "u_700", "u_850", "u_925", "v_10", "v_100", "v_1000", "v_150",
    "v_200", "v_250", "v_300", "v_400", "v_50", "v_500", "v_600", "v_700",
    "v_850", "v_925", "vsw_1", "vsw_2", "w_10", "w_100", "w_1000", "w_150",
    "w_200", "w_250", "w_300", "w_400", "w_50", "w_500", "w_600", "w_700",
    "w_850", "w_925", "z_10", "z_100", "z_1000", "z_150", "z_200", "z_250",
    "z_300", "z_400", "z_50", "z_500", "z_600", "z_700", "z_850", "z_925",
]
