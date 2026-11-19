# Month inference sets: set label of a start date (sourced). MI-S25 = Sep 2025, MI-J25 = Jan 2025 (owner pick 19 Nov 2026).
# Each set has its own Jupiter input folder datasets/mi_<set>_eefo_o320_pf10_l012-240 and output folder outputs/month_inference_<set>.
mi_set_of() { case "${1:0:6}" in 202509) echo s25 ;; 202501) echo j25 ;; *) echo "FATAL no month inference set for date $1" >&2; return 1 ;; esac; }
