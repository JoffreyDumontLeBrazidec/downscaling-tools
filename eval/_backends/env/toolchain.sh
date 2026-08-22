# Shell entry point to the toolchain definitions in eval/config/toolchains.yaml.
#
# Source this from a job script instead of writing your own `module load` lines:
#
#   source /home/ecm5702/dev/downscaling-tools/eval/_backends/env/toolchain.sh
#   use_gptosp
#   ...
#   use_metview
#
# The module lines are rendered by the same Python module the evaluators use, so
# the job scripts and the repo can never drift apart.  Set DS_PYTHON to override
# the interpreter used for rendering.

DS_TOOLCHAIN_ROOT="${DS_TOOLCHAIN_ROOT:-/home/ecm5702/dev/downscaling-tools}"
DS_PYTHON="${DS_PYTHON:-/home/ecm5702/dev/.ds-260612/bin/python}"

use_toolchain() {
  local name="$1"
  local block
  block="$(PYTHONPATH="${DS_TOOLCHAIN_ROOT}:${PYTHONPATH:-}" \
           "${DS_PYTHON}" -m eval._backends.env.toolchain render "${name}")" || {
    echo "FATAL: could not render toolchain '${name}' from ${DS_TOOLCHAIN_ROOT}" >&2
    return 1
  }
  eval "${block}"
}

use_metview() { use_toolchain metview; }
use_gptosp()  { use_toolchain gptosp; }
