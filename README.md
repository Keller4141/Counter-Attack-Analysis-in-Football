# Counter-Attack Detection in International Football

This project identifies and analyses counter-attacks from synchronized event and tracking data. It contains a rule-based definition, a rule-trained XGBoost baseline, a human-guided hybrid XGBoost model, robustness analysis, and DBU-specific analyses for the Danish men's, women's, and U21 national teams.

The submitted project includes `Data/derived/`. These files are the outputs used in the report and allow the main results to be inspected or regenerated without rebuilding snapshots and 180 annotation videos.

## Abstract for the final report
Counter-attacks are an important part of modern football, but they are difficult
to define consistently from event and tracking data because they depend on both
tactical context and temporal development. This project develops a reproducible
framework for detecting and analysing counter-attacks using synchronized event and
tracking data from DBU national-team matches.

First, a rule-based definition was constructed by combining event-based posses-
sion regains with tracking-based spatial measurements and distance-scaled thresholds
for duration, pass count, speed, in-play time, and territorial progression. The defi-
nition was validated against human annotations and showed reasonable agreement
with human football judgment. Second, an XGBoost classifier was trained on engi-
neered event and tracking features to model counter-attack detection from a broader
candidate pool of regain situations. A hybrid human-guided version was then de-
veloped by incorporating majority-vote human labels and agreement-based sample
weights. This hybrid model achieved the strongest agreement with the available
human-labeled evaluation set, outperforming both the rule-based definition and a
rule-trained XGBoost baseline.

Finally, the framework was applied to analyse counter-attacking patterns across
Danish national teams. The results suggest that the Danish men’s senior team did
not primarily lack counter-attacking volume, but rather final outcome quality, as
their attacks less often reached dangerous end locations compared with tournament
benchmarks. The project demonstrates that counter-attacks can be extracted, vali-
dated, and analysed reproducibly through a combination of rule-based logic, human
annotation, and machine learning. However, the findings should be interpreted in
light of limitations related to the small human-labeled evaluation set, event annota-
tion dependency, and the difficulty of capturing tactical intention from observable
data alone.

## Setup

Run commands from the project root. Create a Python environment with:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

The annotation inputs are stored in:

```text
Data/annotations/ANNOTATION_SET.docx
Data/annotations/Kontra vurdering (1).docx
Data/annotations/validation_48/reference_snapshots/
```
The raw DBU data is private and is send seperately to those it belong to:
For running the project, put the DBU data in the following folder: Data:
Data/events.csv
Data/H_EURO2024/
Data/Q_EURO2025/
Data/U21_EURO2025/

This project is also able to run with other football data related to the same format as DBU's. For data that is not the exact same, some changes in the code may be implemented.

## Recommended Execution Order

Because the submitted project already contains the annotation clips, snapshots, and their metadata in `Data/derived/`, the following order reproduces the central model and report results without regenerating visual material.

1. `counterattack_pipeline.ipynb`  
   Applies the final rule-based counter-attack definition. `tracking_direction.py` is imported automatically to determine attacking direction.

2. `build_regain_candidates.py`  
   Builds the model-ready table of regain candidates and event/tracking features.

3. `train_counterattack_detector.py`  
   Trains the rule-labeled XGBoost baseline, performs gain-based feature ranking, and fits the selected top-30 model.

4. `analyze_inter_rater_agreement.py`  
   Reads the human annotations and calculates agreement statistics and Fleiss' kappa.

5. `train_counterattack_detector_human_guided_eval.py`  
   Trains the baseline and human-guided hybrid models with the same features, evaluates them on the fixed 48-case human test set, and scores all regain candidates for the DBU analysis.

6. `counterattack_validation_current_final.ipynb`  
   Produces the report-facing comparison between the final rule-based definition, baseline XGBoost, and hybrid XGBoost.

7. `resampling_human_guided_robustness.py`  
   Runs the 100 repeated human-label split perturbations and the paired permutation tests.

8. `make_resampling_delta_f1_histogram.py`  
   Creates the histogram of paired hybrid-minus-baseline F1 differences.

9. `build_dbu_mens_hybrid_analysis.py`  
   Builds the DBU benchmark tables for the men's, women's, and U21 datasets.

10. `analyze_dbu_mens_shot_profile.py`  
    Builds the shot versus non-shot outcome feature profiles for all three datasets.

11. `dbu_mens_hybrid_analysis.ipynb`  
    Displays the final DBU benchmark and feature-profile tables used in the report.

`list_unique_events.ipynb` is an exploratory data-inspection notebook and is not required by downstream files.

## Report Results

The principal outputs presented in the report can be found here:

| Report result | File or directory |
|---|---|
| Final rule-based counter-attacks | `Data/derived/counterattack_attempts_balanced_v2_scaled_tracking_coords.csv` |
| Rule-based profile summaries | `Data/derived/counterattack_profile_summary_tracking_coords.csv` and `Data/derived/counterattack_start_zone_tracking_coords.csv` |
| Model-ready regain candidates | `Data/derived/ai_counterattack_detection/regain_candidates_labeled.csv` |
| Baseline model metrics | `Data/derived/ai_counterattack_detection/model_metrics.json` |
| Gain-based feature importance | `Data/derived/ai_counterattack_detection/feature_importance.csv` |
| Inter-rater agreement | `Data/derived/inter_rater_agreement/combined_summary.csv` and the JSON summaries in the same directory |
| Fixed 48-case model comparison | `Data/derived/validation_legacy_broad_pool/current_comparison_metrics.csv` |
| Fixed-test labels and predictions | `Data/derived/validation_legacy_broad_pool/current_comparison_docx_labels.csv` |
| Split perturbation summary | `Data/derived/ai_counterattack_detection/human_resampling_robustness/resampling_summary.csv` |
| Paired F1 differences and p-values | `Data/derived/ai_counterattack_detection/human_resampling_robustness/resampling_paired_differences.csv` |
| Split-level robustness results | `Data/derived/ai_counterattack_detection/human_resampling_robustness/resampling_iteration_metrics.csv` |
| Robustness histogram | `figures/hybrid_baseline_delta_f1_histogram.png` and `.pdf` |
| DBU benchmark tables | `Data/derived/dbu_mens_hybrid_analysis/denmark_vs_hybrid_benchmarks_*.csv` |
| Team-level DBU summaries | `Data/derived/dbu_mens_hybrid_analysis/team_hybrid_counterattack_summary_*.csv` |
| DBU shot/non-shot feature profiles | `Data/derived/dbu_mens_hybrid_analysis/shot_profile/shot_vs_nonshot_feature_profile_*.csv` |

The submitted outputs reproduce the main reported values, including 746 rule-based counter-attacks, baseline test F1 of 0.886, fixed human-test F1 scores of 0.792 for the baseline and 0.815 for the hybrid model, and mean split-perturbation F1 scores of 0.673 and 0.742 respectively.

## Full Regeneration of Visual Material

The following additional steps are only necessary if `Data/derived/` is removed or the snapshots and annotation videos must be recreated from raw data.

After running `counterattack_pipeline.ipynb`, `build_regain_candidates.py`, and `train_counterattack_detector.py`:

1. Run `build_validation_legacy_broad_pool.py` to recreate the broad validation pool and its metadata CSV files.
2. Run `make_validation_candidate_snapshots.py` to recreate the validation snapshots.
3. Run `make_human_annotation_set.py` to recreate the six annotation groups, 180 MP4 clips, and `annotation_selection.csv`.
4. Continue from `analyze_inter_rater_agreement.py` in the recommended execution order above.

The visual generation steps are considerably slower than the model scripts because they repeatedly read tracking data and render individual images or videos. The fixed 48-case human validation images are stored separately in `Data/annotations/validation_48/reference_snapshots/` because they are annotation inputs rather than generated report results.

## Verified Outputs

The central execution order was tested on 11 June 2026. All required scripts and notebooks completed without runtime errors. Snapshot and MP4 generators were intentionally excluded from that verification because their completed outputs are included in `Data/derived/`.
