# Study registry

Study scripts are not library modules, so lack of import callers is not evidence
that they are obsolete. This registry records their role and prevents accidental
deletion of research provenance.

## Production campaign

- `real_noise_event.py`: one event/class task for the real-noise campaign.
- `real_noise_io.py`: shared campaign bundle and manifest I/O.
- `collect_results.py`: merge per-event results and report missing outcomes.
- `real_noise_aggregate.py`: aggregate event rows into campaign summaries.
- `paired_network_audit.py`: construct complete paired detector cohorts.
- `real_noise_plots.py`: generate the real-noise manuscript figures and table.

The production entry point is `slurm/submit_campaign.sh`. Large bundles, chains,
and event products belong outside Git; committed summaries must retain enough
metadata to identify their campaign and inputs.

## Validation and benchmarks

- `bench_sky_sampling.py`: fixed-sky versus sampled-sky NUTS benchmark.
- `chisq_baseline.py`: reweighted matched-filter baseline.
- `fitting_factor.py`: held-out CCSN decoder fitting factors.
- `morphz_validation.py`: analytic validation of the morphZ estimator.
- `noise_scale_marginal.py`: Whittle convention and noise-scale validation.
- `plot_multimodality_diagnostic.py`: per-chain multimodality diagnostics.
- `simulated_design_psd.py`: design-PSD simulation workflow.
- `snr_vs_odds_roc.py`: single-detector design-PSD ROC study.
- `snr_vs_odds_roc_coherent.py`: coherent multi-detector ROC study.

## Manuscript figures and calibration

- `plot_waveform_reconstruction.py`: Starccato/BayesWave reconstruction figure.
- `pp_predictive_fig.py`: posterior-predictive manuscript figure.
- `pp_test.py`: P-P calibration and combined P-P plot.

## Small workflow wrapper

- `simulation_study.py`: run the three standard scenarios from an explicit
  trigger CSV and analysis configuration.

## Legacy candidates retained for provenance

- `bilby_sim.py`: pre-production Bilby prototype; executes work at module scope.
- `psd_plot.py`: exploratory PSD plotting script; executes work at module scope.
- `v2.py`: obsolete analysis-call prototype with source-tree-specific inputs.

These three files should not be used for new results. Delete them only after
confirming that no historical result or manuscript statement depends on their
exact implementation.
