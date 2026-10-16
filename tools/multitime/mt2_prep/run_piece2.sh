#!/bin/bash
S=/home/ecm5702/scratch/sandbox/20261016-multitime-e2
PY=/home/ecm5702/hpcperm/sandbox/20260917-vera-spatial-downscaler/.venv-x86_64/bin/python
export PYTHONPATH=$S/code/anemoi-core/models/src:$S/code/anemoi-core/training/src:$S/code/anemoi-core/graphs/src
mkdir -p $S/assets
date -u
$PY -c "import anemoi.training, anemoi.models, anemoi.graphs; print(anemoi.training.__file__, anemoi.models.__file__, anemoi.graphs.__file__)"
VARS=10u,10v,2d,2t,msl,q_100,q_1000,q_200,q_300,q_400,q_50,q_500,q_700,q_850,q_925,skt,sp,t_100,t_1000,t_200,t_300,t_400,t_50,t_500,t_700,t_850,t_925,tcw,u_100,u_1000,u_200,u_300,u_400,u_50,u_500,u_700,u_850,u_925,v_100,v_1000,v_200,v_300,v_400,v_50,v_500,v_700,v_850,v_925,w_100,w_1000,w_200,w_300,w_400,w_50,w_500,w_700,w_850,w_925,z_100,z_1000,z_200,z_300,z_400,z_50,z_500,z_700,z_850,z_925
$PY $S/scripts/convert_residual_stats_to_npz.py /ec/res4/hpcperm/ecm5702/data/residuals_statistics/o1280_dict_0_72.npy $S/assets/o1280_dict_0_72.npz \
  --check-variables $VARS,cp,tp --check-residual-variables $VARS \
  --compare /home/ecm5702/scratch/sandbox/20261007-pl25-ac/assets/o1280_dict_0_72_precip.npy
echo rc=$?
date -u
