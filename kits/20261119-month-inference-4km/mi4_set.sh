# MI4-S26 settings (sourced by every script of this kit; settings live here, never in exported env).
# 4 km lane month inference input set: September 2026, od enfo pf 1-10, 00 UTC, native O1280, FULL GLOBE, leads 12..240 h / 6 h.
MI4_SET=s26
MI4_A=/ec/res4/scratch/ecm5702/agent-work/20261119-mi4k-s26            # Atos work folder (grib/, stores/, keep/, state/, logs/)
MI4_K=$MI4_A/dt/kits/20261119-month-inference-4km                      # this kit (downscaling-tools branch claude/project-thread-fk96ie)
MI4_META=$MI4_A/ref/in_lres_o1280_meta.zarr                            # vars, lat/lon, mean/stdev of Jupiter o1280_o2560_global/in_lres_o1280.zarr
MI4_JD=/e/data1/jureap-data/ecmwf/users/jdumont/datasets/mi4_s26_enfo_o1280_pf10_l012-240   # Jupiter destination
MI4_DATES=$MI4_K/dates_s26.txt
MI4_PAR=4          # concurrent MARS clients per retrieval job (pressure-level step groups); read at job start
MI4_CAP=4          # max dates on Atos between retrieval start and verified copy to Jupiter (disk cap ~4 x 430 GB)
MI4_LANES=3        # parallel retrieval->build lanes (used by mi4_chain_submit.sh at submit time only)
