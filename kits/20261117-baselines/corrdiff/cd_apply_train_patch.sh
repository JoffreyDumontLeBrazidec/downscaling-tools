#!/bin/bash
# Write examples/weather/corrdiff/train_cd.py = the pristine v2.2.2 train.py + train_cd_v2.2.2.patch (EMA, regression
# crops, data-wait time; code in cd_train_ext.py). train.py itself stays untouched (checked with git). Idempotent.
#   CD_REPO=<physicsnemo v2.2.2 checkout> CD_KIT=<kit folder> bash cd_apply_train_patch.sh
set -euo pipefail
: "${CD_REPO:?}" "${CD_KIT:?}"
EX=$CD_REPO/examples/weather/corrdiff
[[ $(git -C $CD_REPO rev-parse HEAD) == 072465a1a56817f180f64dee8a4069a1612b2d9e ]] || { echo "PATCH_FAIL $CD_REPO is not tag v2.2.2"; exit 1; }
git -C $CD_REPO diff --quiet -- examples/weather/corrdiff/train.py || { echo "PATCH_FAIL train.py is modified"; exit 1; }
patch -s -o $EX/train_cd.py.new $EX/train.py < $CD_KIT/train_cd_v2.2.2.patch
mv $EX/train_cd.py.new $EX/train_cd.py
grep -q CD_KIT_PATCH $EX/train_cd.py
echo "PATCH_OK $EX/train_cd.py sha256 $(sha256sum $EX/train_cd.py | cut -c1-16) from $(sha256sum $CD_KIT/train_cd_v2.2.2.patch | cut -c1-16)"
