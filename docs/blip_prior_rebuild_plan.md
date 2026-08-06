# Rebuilding the blip prior: implementation plan

Status: proposed 2026-08-06. Supersedes nothing yet; the production campaign
still uses the gengli-derived `blip` decoder with a pinned transient time.

## Why

Two coupled defects, both measured (see `manuscript/PRD_REFEREE_NOTES.md`):

1. **Timing asymmetry.** The signal hypothesis sat at `t_c + dt_i` while the
   detector-local glitch sat at the trigger. Fixed: `--time-marginal`
   (implemented, validated to 1e-13, 6 tests).
2. **Glitch-prior population mismatch.** The blip decoder is a faithful model
   of *L1 O2* blips (training f50 = 85 Hz; L1 O2 catalog median peak frequency
   89.7 Hz) applied to *O3b* data, where the L1 median is 269.3 Hz. L1's blip
   population tripled in frequency between runs. This plan fixes defect 2.

With (1) fixed and (2) outstanding, event 0's blip flips to a "signal" at
single detector (`lnO = +906`). The H1-L1 configuration still rejects it, and
more strongly than before (`lnO = -1439.6` vs `-412.6` pinned), so coherence is
the load-bearing mechanism and is *not* at risk. What is at risk is every
single-detector number and the claim that rejection is morphological.

## Two design questions, answered

### Can we avoid running BayesWave?

**Probably yes, subject to one validation gate (Phase 0c).**

BayesWave buys denoising: it separates glitch waveform from noise. The cheap
route buys the same thing with PCA truncation, and the noise arithmetic is
favourable. Whitened noise spreads uniformly over all 512 core dimensions;
truncating to `k` principal components retains `k/512` of the noise power while
retaining most of the signal (measured: top-5 components carry 86% of the
population variance across real cores). At `k = 30` that is a **~4x amplitude
SNR gain** (`sqrt(30/512) = 0.24`), which turns a moderate in-band SNR-20 blip
into an effectively SNR-80 shape measurement.

The honest caveats, and why the gate exists:
- BayesWave is the established precedent — gengli itself was trained on
  BayesWave reconstructions — so a referee may ask. The answer must be a
  measurement, not an argument.
- BayesWave models non-Gaussian noise; PCA assumes the noise is Gaussian in
  the whitened basis.
- Against it: BayesWave costs ~63 min/event here and its evidence uncertainty
  exceeded 10 nats for 48% of runs in the v0.44 campaign. For *waveform
  reconstruction* (not evidence) it is better behaved, but it is not free.

**Gate (Phase 0c):** inject known clean waveforms into real off-source detector
noise at controlled in-band SNR, run the full extract -> PCA -> un-whiten
pipeline, and measure the match to truth. Ship the cheap route if the recovered
shape matches at >= 0.98 for in-band SNR >= 20, and note the SNR floor below
which it degrades. If it fails, fall back to Phase 3.

**Results, 2026-08-06 (two bugs fixed first -- see Phase 0b section for both;
`unwhiten` peak-alignment and `roll_off=0.1` for wide-band whitening).**

- *Narrow band (300-800 Hz), n=19 loud cores*: median match 0.975 at SNR>=10,
  flat with SNR -- near-pass, gap looked like small-sample leave-one-out cost.
- *Wide band (30-1024 Hz), n=72 cores spanning the full peak-frequency range,
  both detectors*: plateaus at **0.79 at k=70** (out of 71 available "other"
  cores -- i.e. using nearly the full-rank basis) and is STILL RISING. This is
  not a "need more components" ceiling; the pooled population (peak frequency
  90-1200 Hz) contains structurally different time-domain shapes that a single
  shared linear basis of this size cannot span, however large `k` gets.
- *Frequency-stratified (basis fit WITHIN <250 Hz and >=250 Hz groups
  separately, matching Phase 1's actual gengli-mix design)*: see
  `studies/blip_denoiser_injection_test_stratified.py` for the direct test of
  whether stratifying (rather than pooling) recovers something closer to the
  narrow-band ceiling.

**RESOLVED 2026-08-06 — the wide band was the bug, not the population.**
Measured the intrinsic ceiling of the extraction itself (`unwhiten(extract_core(x))`
vs band-limited truth, post-alignment-fix):

| band | half_width | extraction match |
|---|---|---|
| 300-800 Hz | 256 | **0.998** |
| 300-800 Hz | 512 | 0.994 |
| 30-1024 Hz | 256 | **0.247** |
| 30-1024 Hz | 512 | 0.395 |
| 30-1024 Hz | 1024 | 0.328 |

A 30 Hz high-pass has an impulse response seconds long, so a 512-sample
(125 ms) core cannot contain its whitened representation; widening the core
does not help because the response is not a millisecond-scale effect. Every
wide-band Phase 0c number was therefore capped by an extraction achieving only
0.25-0.40 fidelity. The 0.79 pooled plateau and the 0.44/0.68 frequency-split
were artifacts of that ceiling, NOT statements about blip population
diversity. Retracted.

The wide extraction band was a self-inflicted design choice from this plan's
own Phase 1 step 1 ("extract wide so the model is not structurally
band-limited") -- speculative future-proofing for a band ablation that may
never happen. It is dropped.

**Narrow-band verification on the FULL stratified population** (54 of 85 blips
have in-band peak/floor >= 10, spanning peak_f 176-1192 Hz): leave-one-out
match 0.894 (k=3), 0.897 (k=5), 0.907 (k=15) -- essentially FLAT in k, the
signature of a genuinely low-dimensional population plus a noise floor rather
than an under-resourced basis. Consistent with z=5.

**DECISIONS:**
1. **Extract in 300-800 Hz** (the analysis band). Gate passes: extraction
   0.998, injection-recovery 0.975 at SNR>=10. **BayesWave (Phase 3) is not
   needed.**
2. **Do NOT mix gengli.** The Phase 0b mixing verdict was measured in the wide
   band, which is no longer used. In 300-800 Hz gengli's in-band power
   fraction is 0.0001 (median over prior draws AND training waveforms) -- it
   contributes nothing where the likelihood looks. Phase 0b's result stands as
   a statement about wide-band morphology but no longer bears on this choice.
3. Train blip-v2 on **real O3a cores only**, evaluate on O3b.

Consequence to carry into Phase 2: blips whose power sits below the analysis
band are invisible in-band by construction (a peak_f=176 Hz blip has a
300-800 Hz whitened peak of 0.03 sigma). They will be selected as *noise*,
not as glitches. That is honest -- the analysis genuinely cannot see them --
but it means the blip->noise cell of the confusion matrix is expected to stay
populated, and should be reported as a band-coverage limitation rather than a
model failure.

### Should we mix gengli outputs into the training set?

**Test before deciding — and note we have never actually tested gengli on the
blips it models.** Our fitting-factor study used catalog indices 0-31, i.e. the
loudest blips, whose median peak frequency is 630 Hz; only **9%** of that
sample falls inside gengli's own [30, 250] Hz range. The conclusion "the blip
decoder underfits real blips" is therefore established only *outside* gengli's
domain. 531 L1 O3b blips (45% of the catalog) peak below 250 Hz and have never
been probed.

**Gate (Phase 0b) — DONE, 2026-08-06, both detectors independently:**
wide-band (30-1024 Hz, `roll_off=0.1`) fitting factor of gengli vs ccsne on the
full stratified sample (47 L1 + 38 H1 real O3b blips), binned by GravitySpy
`peak_frequency`, with the +-25 ms shift search restored (dropping it, tried
first, collapses FF for any residual sub-sample misalignment). The relative
comparison is what decides this, not an absolute FF threshold -- a properly
SNR-diverse stratified sample scores much lower in absolute terms (~0.15-0.3)
than a loud-only sample (~0.5-0.9) for BOTH decoders, so an absolute bar like
"FF>=0.9" was tried and retracted as miscalibrated.

| peak_f bin | L1 (ccsne/blip/wins) | H1 (ccsne/blip/wins) |
|---|---|---|
| 0-150 Hz | 0.116/0.128/7-8 | 0.156/0.181/8-8 |
| 150-250 Hz | 0.180/0.187/7-8 | 0.250/0.258/6-8 |
| 250-350 Hz | 0.158/0.145/5-8 | 0.265/0.239/2-8 |
| 350-550 Hz | 0.194/0.191/4-8 | 0.263/0.199/1-8 |
| 550-900 Hz | 0.129/0.128/2-7 | 0.228/0.175/1-6 |
| 900+ Hz | 0.286/0.165/0-8 | -- |

**Below 250 Hz: gengli beats ccsne on 14/16 events on BOTH detectors
independently (87.5% each)** -- a strong, cross-validated result. Above 250 Hz
the detectors diverge in a physically sensible way: H1 shows a sharp crossover
to ccsne (matches H1's tighter, lower-frequency population); L1 stays roughly
tied out to ~900 Hz (matches L1's broader, higher-frequency population,
median 269 Hz) and only loses clearly above 900 Hz.

**Decision: mix gengli in, gated at `peak_frequency < 250 Hz`, weighted by
population share** (~45% L1, ~71% H1, from the full-catalog census in
`docs/blip_prior_rebuild_plan.md`'s population table) rather than raw draw
count -- 2000 gengli draws against ~400 real cores would otherwise let an O2
population dominate an O3 prior.

Script: `studies/blip_gengli_ff_by_freq.py`. Two bugs found and fixed while
building this (see `studies/blip_core_extract.py` docstrings): (1) `unwhiten`
padded a core at segment-center instead of the actual peak index -- any
off-center peak silently corrupted reconstruction (round-trip match
0.008 -> 0.93 after the fix); (2) wide-band FFT whitening on a 4 s segment
produces a boundary artifact (circular convolution wrapping a 30 Hz
high-pass's long impulse response) that the default Tukey taper
(`roll_off=0.01`, ~20 ms/edge) cannot suppress -- `roll_off=0.1` (~400 ms/edge)
fixes it (measured: edge artifact 7.4sigma -> 0.04sigma, true peak unchanged).
Both bugs contaminated an earlier same-session pass at this gate; that pass's
numbers were discarded, not corrected in place.

## Phase 0 — diagnostics that fix the design (~half a day, no cluster)

| step | what | decides |
|---|---|---|
| 0a | Stratified fetch: ~60 O3b blips per detector, sampled across the 5 peak-frequency bins x SNR terciles | sample for 0b/0c |
| 0b | FF(gengli), FF(ccsne) vs peak frequency on that sample | whether to mix gengli |
| 0c | Injection-recovery of the PCA denoiser at controlled SNR | whether BayesWave is needed |

Stratification bins (L1 O3b, from the catalog's `peak_frequency`):
`[0,150) n=264`, `[150,250) n=267`, `[250,350) n=272`, `[350,550) n=263`,
`[550,900) n=103`.

New script: `studies/blip_core_extract.py` (extraction + PCA + un-whitening,
shared by 0c and Phase 1). Reuses the validated core-extraction logic already
prototyped in this session.

## Phase 1 — cheap route (2-3 days)

Train on **O3a**, evaluate on **O3b** (measured: same-detector O3a-vs-O3b KS
D = 0.086 (L1) / 0.060 (H1) — nearly identical populations; cross-host is
D = 0.345 / 0.268 and would conflate generalization with a population shift).

1. **Curate.** `studies/build_blip_training_set.py`:
   - stratified sample of O3a blips per detector across peak-frequency bins,
     targeting ~400/detector, spanning the SNR range (not just the loud tail,
     because peak frequency correlates with SNR and a loud-only set would be a
     high-frequency set);
   - fetch bundles, estimate the off-source PSD exactly as the analysis does;
   - whiten, locate the peak, cut a 512-sample core;
   - quality cuts: in-band peak/floor >= 10, peak within +-5 ms of the
     catalog trigger, data-quality check on the parent segment;
   - PCA-truncate to `k ~= 30` components (fit on the training set only);
   - **un-whiten each core with its own event PSD** back to raw strain, over a
     wide extraction band (30-1024 Hz, not 300-800) so the model is not
     structurally band-limited and a future band ablation stays possible;
   - **sign-align, then augment with the negated copy.** The glitch amplitude
     is `exp(log_amp) > 0` and cannot flip polarity, so the decoder must span
     both. Sign-aligning first makes the manifold compact (PC1 = 63% of
     variance when aligned; near 0 when not), and augmenting restores both
     polarities at no latent cost.
   - standardize per waveform (zero mean, unit variance), matching
     `TrainValData.load`; write `blip_o3a_real.npz` in the same format.
2. **Train** `blip-v2` with the existing pipeline and the existing z=5,
   capacity-constrained objective. No architecture change: leave-one-out
   linear reconstruction of real cores already reaches match 0.983 with a
   5-dimensional basis, so z=5 is the right size and this is a data problem.
3. **Validate before any inference**, on held-out **O3b**:
   - FF(blip-v2) vs real O3b blips, by peak-frequency bin — target: beats
     FF(ccsne) (currently 0.615 vs 0.565 against it) across all bins;
   - FF(blip-v2) vs **held-out CCSN injections** — must stay *low*. A glitch
     model good at fitting clicks will also fit bounce signals; this is the
     headline risk and must be measured here, before the pilot;
   - prior-predictive spectra vs the O3b population (the honest replacement
     for the current Figure 1 panel).

## Phase 2 — pilot (~1 day compute)

~30 trigger times x 3 classes x {1-det, H1-L1}, with `--time-marginal` and
blip-v2. Report: three-class confusion, `lnO` separation per class, and
**signal recall** — the number that decides whether the operating point moved.
Compare against the same events under the production configuration.

Gate: proceed to a campaign only if blips are rejected morphologically at one
detector (`lnZ_G > lnZ_S`) *and* signal recall stays above ~95%.

## Phase 3 — BayesWave route (only if Phase 0c or Phase 1 validation fails)

OzSTAR, data already local. Reuses `slurm/bayeswave_*.sh`.
- ~300 O3a blips per detector, **full band** (not 300-800) so the
  high-frequency content that gengli lacks is actually reconstructed;
- take the posterior median reconstruction per event (optionally a few draws
  per event as augmentation);
- feed the same Phase 1 step 2 onward.

Cost: ~300 events x ~1 h x 2 detectors, embarrassingly parallel. The
methodological payoff is that it reproduces gengli's own construction on the
right population, which is the easiest thing to defend in review.

## Phase 4 — campaign and manuscript

1. Full campaign rerun with `--time-marginal` + blip-v2, fresh `CAMPAIGN_ID`.
2. Manuscript reframe:
   - coherence becomes the primary result (it strengthens: `-412.6 -> -1439.6`
     on event 0);
   - the single-detector arm stays, reported honestly as the ablation showing
     morphology alone is insufficient — dropping it would remove the only
     evidence for the coherence gain and would be selective reporting;
   - Figure 1's premise is rewritten around the **O3** blip population, not
     gengli's O2 one;
   - the SNR* baseline comparison becomes fair once lnO also has time freedom
     (SNR* is maximized over +-0.1 s).

## Risks

- **Signal recall.** The current 97.6%/99.2% is partly *because* `H_G` is weak.
  A competent glitch model will eat some injections. Measured in Phase 1
  validation and Phase 2, not assumed.
- **Circularity.** Train O3a, evaluate O3b, never mix. Do not tune anything on
  O3b outcomes.
- **Loud-blip bias.** Peak frequency correlates with SNR, so any "top-N by SNR"
  sample is a high-frequency sample. Stratify explicitly (this is the mistake
  that biased our first FF study).
- **PCA denoiser assumes Gaussian whitened noise.** Phase 0c is the gate; the
  low-frequency bins are where it is most likely to fail, and where mixing
  gengli would help most.
- **Extraction band vs analysis band.** Extract wide (30-1024 Hz) so the model
  is not structurally band-limited, even though the analysis currently uses
  300-800 Hz.
