# Month inference sets: set label of a start date (sourced). MI-S25 = Sep 2025, MI-J25 = Jan 2025, MI-JUL25 = Jul 2025 (19 Nov 2026).
# IFS cycle 48r1 sets (24 Nov 2026): MI-J24 = Jan 2024, MI-JUL24 = Jul 2024, MI-S24 = Sep 2024 (same months as the 2025 sets, one year earlier).
# Each set has its own Jupiter input folder datasets/mi_<set>_eefo_o320_pf10_l012-240 and output folder outputs/month_inference_<set>.
mi_set_of() { case "${1:0:6}" in 202509) echo s25 ;; 202501) echo j25 ;; 202507) echo jul25 ;; 202401) echo j24 ;; 202407) echo jul24 ;; 202409) echo s24 ;; *) echo "FATAL no month inference set for date $1" >&2; return 1 ;; esac; }
