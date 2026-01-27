Kenwood KAC-823 baseline (auto-scaled THD sweeps)
=================================================

Sweep conditions:
- Bridge mode, 4 Ω dummy load, Tek P2220 10x probe on CH1 (amp output) & CH3 (FY split)
- Generator amplitudes: 0.2, 0.3, 0.4, 0.5 Vpp (FY3200S)
- THD captured via Tek math trace (CH1-CH3)
- Automatic vertical scaling enabled with `--scope-auto-scale CH1=13,CH3=1 --scope-auto-scale-margin 0.8`
- Gold calibration applied in *_auto_gold.csv runs via packaged gold reference

Files:
- thd_0p{2,3,4,5}_auto_CH1-13_margin0p8.csv  -> raw auto-scaled sweeps
- thd_0p{2,3,4,5}_auto_gold.csv               -> gold-calibrated counterparts
- thd_auto_summary.csv                        -> raw statistics (THD mean/min/max, THD@1k, Vrms@1k)
- thd_auto_gold_summary.csv                   -> gold-calibrated statistics
- thd_auto_merged_summary.csv                 -> side-by-side comparison (incl. deltas)
- thd_auto_overlay.png                        -> raw THD vs frequency overlay (0.2–0.5 Vpp)
- thd_auto_gold_overlay.png                   -> gold-cal overlay
- thd_vs_amp_raw_vs_gold.png                  -> THD@1k vs amplitude (raw vs gold)
- vrms_vs_amp_raw_vs_gold.png                 -> Vrms@1k vs amplitude (raw vs gold)

Use this directory as the reference dataset before changing loads, supply voltage, or probe configuration.
