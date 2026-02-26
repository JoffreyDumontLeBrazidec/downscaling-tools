#!/bin/bash
#SBATCH --job-name=tc_get_j24v
#SBATCH --qos=nf
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/home/ecm5702/perm/eval/j24v/logs/tc_get_j24v_%j.out

set -euo pipefail
mkdir -p /home/ecm5702/perm/eval/j24v/logs

STREAM="ENFO"
GRID="O320"
EXPID="j24v"
STEP="24/to/120/by/24"
PARAMS="151.128/165.128/166.128/167.128"
NUMBER="1/to/10/by/1"
TIMES="0000"
CLASS="RD"
TYPE="PF"
LEVEL="SFC"
DATABASE="FDB"
YEAR="2023"
MONTH="08"

fetch_event () {
  local event="$1"
  local area="$2"
  shift 2
  local days=("$@")
  local dir="/home/ecm5702/hpcperm/data/tc/${event}"
  mkdir -p "${dir}"

  for d in "${days[@]}"; do
    date="${YEAR}${MONTH}${d}"
    target="${dir}/surface_pf_${STREAM}_${GRID}_${EXPID}_${date}.grib"
    if [ -s "${target}" ]; then
      echo "[SKIP] ${target} already exists"
      continue
    fi
    req="/tmp/tc_${event}_${date}.mars"
    cat > "${req}" <<MARS
RETRIEVE,
  CLASS      = ${CLASS},
  TYPE       = ${TYPE},
  STREAM     = ${STREAM},
  EXPVER     = ${EXPID},
  LEVTYPE    = ${LEVEL},
  AREA       = ${area},
  PARAM      = ${PARAMS},
  DATE       = ${date},
  TIME       = ${TIMES},
  NUMBER     = ${NUMBER},
  GRID       = ${GRID},
  DATABASE   = ${DATABASE},
  STEP       = ${STEP},
  TARGET     = "${target}"
MARS
    mars "${req}"
    rm -f "${req}"
  done
}

fetch_event idalia "40/-100/10/-70" 26 27 28 29 30
fetch_event hilary "35/-125/0/-95" 16 17 18 19 20
fetch_event franklin "40/-80/10/-50" 20 21 22 23 24 25 26 27 28 29 30
fetch_event fernanda "30/-135/0/-105" 12 13 14 15 16 17
fetch_event dora "25/175/5/-105" 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17
