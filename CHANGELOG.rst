.. _changelog:

=========
CHANGELOG
=========


Unreleased
==========

NUTS initialization
-------------------

* Add a fast Sobol multistart MAP initializer for the Starccato latent
  posterior. The default NUTS and BCR workflows now screen 128 prior-scaled
  latent starts, refine the four highest-density distinct basins, and pass the
  best point to NumPyro's ``init_to_value`` strategy.
* Persist every optimization attempt and clustered-basin diagnostic in
  ``map_initialization.json``. The workflow warns when the winning basin was
  reached by only one broad start or when a secondary basin lies within five
  log-density units. These checks diagnose, but do not eliminate, posterior
  multimodality.
* For multi-chain runs, initialize NUTS chains across all screened basins
  within five log-density units of the best solution (round-robin when there
  are more chains than basins). CCSNE chains use one compiled vectorized
  sampler; the longer-running blip likelihood uses the memory-safer sequential
  chain map. Persistent mode separation therefore becomes a visible
  cross-chain diagnostic.
* Require at least two chains in high-level NUTS workflows. Abort before any
  morphZ or fallback evidence calculation when the maximum R-hat exceeds 1.05,
  no finite R-hat is available, or any divergence is present. Values above
  1.01 remain a publication-quality warning.


.. _changelog-v0.0.3:

v0.0.3 (2026-07-01)
===================

Bug Fixes
---------

* fix: correct BCR evidence normalization and coherent injection

The single-detector BCR pipeline produced no valid log-evidence ratios:
every signal/glitch evidence came back NaN and the noise reference used
an inconsistent normalisation. Three issues are addressed here.

1. Noise evidence normalisation (main.py)
   The JIM transient likelihood is noise-relative (<h|d> - <h|h>/2, with
   the -<d|d>/2 and log(2*pi*PSD) constants dropped), so the morphZ /
   nested signal and glitch evidences are already ratios relative to
   noise. The per-detector constants cancel exactly in the BCR, so the
   self-consistent noise reference is logZ_noise = 0, not -<d|d>/2.
   Using -<d|d>/2 (a large negative number) suppressed the noise
   hypothesis and gave the wrong BCR sign for noise and glitch data.

2. morphZ usage (main.py)
   morph_type="pair" requires a precomputed params_MI.json that the code
   never generated, so morphZ raised FileNotFoundError on every call and
   the broad except returned NaN silently. Switch to morph_type="indep"
   (auto-computes its bandwidths) and surface the exception instead of
   hiding it.

3. Coherent injection (run_event.py)
   inject_signal added the raw sky-frame waveform identically to every
   detector, but the recovery model projects through each detector's
   response (antenna pattern + time delay). The response time-delay
   reference depends on the data epoch, which is only set on prepared
   detectors; a bare detector clone places the signal at the wrong time
   and the coherent template cannot match it. Project the injection with
   detectors prepared identically to the recovery, and preserve the
   bundle's root-level metadata (trigger_time) when writing the injected
   bundle.

With these fixes the BCR has the expected sign in all three scenarios:
noise ~ 0, coherent injection strongly positive, single-detector blip
glitch strongly negative; morphZ-indep and nested-sampling evidences
agree. (`6972867`_)

Unknown
-------

* add studies (`fe29a51`_)

* Merge pull request #2 from starccato/feat/morphz-evidence-validation

update lnz computation (`b390a89`_)

* update lnz computation (`f42e5ce`_)

* Merge pull request #1 from starccato/fix/bcr-evidence-and-coherent-injection

fix: correct BCR evidence normalization and coherent injection (`bb4502d`_)

* add JIM (`66a2c8d`_)

* add a bilby si and JIM likelihood testing (`fa05add`_)

* add new trigger getter (`3920a9d`_)

* add more testing (`c24590a`_)

* add datacheck (`58b7b13`_)

* fix setup (`a3f9412`_)

* add logs (`0be96a7`_)

* make multidector analysis work (`613db9f`_)

* add analysis configs (`dae7075`_)

* refactoring analysis (`30da722`_)

* recache ozstar files (`586bd26`_)

* add cli docstrings (`221c933`_)

* Merge branch 'main' of github.com:starccato/starccato_lvk (`156a785`_)

* fix signal injection (`94bba64`_)

* fix errors for out of seg (`2e5968d`_)

* Merge branch 'data_prep_hacking' into main (`d015d2a`_)

* hacking on data loaindg (`9ed82d5`_)

* data prep hacking (`3dbda29`_)

* add notes for study (`47fdf0b`_)

* major refactoring (`ba00cf0`_)

* major refactoring (`39405c7`_)

* Merge branch 'main' of github.com:starccato/starccato_lvk (`eeba077`_)

.. _6972867: https://github.com/starccato/starccato_lvk/commit/6972867752aeab4fa7b4760596d535dbe3153366
.. _fe29a51: https://github.com/starccato/starccato_lvk/commit/fe29a51e9d463f8e84e4dd2aae307c5dee6c5c57
.. _b390a89: https://github.com/starccato/starccato_lvk/commit/b390a8919a7bda8077a66455c89f58653638aa97
.. _f42e5ce: https://github.com/starccato/starccato_lvk/commit/f42e5ce8888902dcb5879e88420e2be8166d1769
.. _bb4502d: https://github.com/starccato/starccato_lvk/commit/bb4502d29435304f5450a1f38fb85d188c055b27
.. _66a2c8d: https://github.com/starccato/starccato_lvk/commit/66a2c8dfd17833a887b69099a1cad94d19d4a645
.. _fa05add: https://github.com/starccato/starccato_lvk/commit/fa05adda6f9b6566989bb1c0b76f5bd630ad8c50
.. _3920a9d: https://github.com/starccato/starccato_lvk/commit/3920a9d8ff3079ac803817e6bd0492c5011d2215
.. _c24590a: https://github.com/starccato/starccato_lvk/commit/c24590a76bd986284c26c3f3f4f029b39bacffc2
.. _58b7b13: https://github.com/starccato/starccato_lvk/commit/58b7b133391a937067971bc9e2806ea53bc0297c
.. _a3f9412: https://github.com/starccato/starccato_lvk/commit/a3f94128451904e9dd7f1d219562a7f0b7075f3c
.. _0be96a7: https://github.com/starccato/starccato_lvk/commit/0be96a70d427d19882b6b0829479872ca88b36f1
.. _613db9f: https://github.com/starccato/starccato_lvk/commit/613db9f64bae28e0c7f2791fcac6c7b2cabee43a
.. _dae7075: https://github.com/starccato/starccato_lvk/commit/dae707587804c74b4ded8060163ad326597bed35
.. _30da722: https://github.com/starccato/starccato_lvk/commit/30da7229abcb230fbf322b9818ca567dec4a5b84
.. _586bd26: https://github.com/starccato/starccato_lvk/commit/586bd2638d73f186df33954ca4a6de3e1bb4cd9e
.. _221c933: https://github.com/starccato/starccato_lvk/commit/221c933b7b5c21a20a3162d97dfe35a69356438b
.. _156a785: https://github.com/starccato/starccato_lvk/commit/156a7853e152d3a4db6d434bbe3fcaf72e900de6
.. _94bba64: https://github.com/starccato/starccato_lvk/commit/94bba64a069566784bfb5d53e8879151a9a77462
.. _2e5968d: https://github.com/starccato/starccato_lvk/commit/2e5968de8cbc0b1cae9264bbd3209c662b6935b9
.. _d015d2a: https://github.com/starccato/starccato_lvk/commit/d015d2a43ffbff57f539257b008797097dcdef95
.. _9ed82d5: https://github.com/starccato/starccato_lvk/commit/9ed82d538153e7b32341d0a453bd682e049138ec
.. _3dbda29: https://github.com/starccato/starccato_lvk/commit/3dbda29af916536e7fdf3ef2de6930df7445476f
.. _47fdf0b: https://github.com/starccato/starccato_lvk/commit/47fdf0b90e19f942beb2246c944543659d8eb61e
.. _ba00cf0: https://github.com/starccato/starccato_lvk/commit/ba00cf0335f27676763ef985ab6764982ece05da
.. _39405c7: https://github.com/starccato/starccato_lvk/commit/39405c7e598466666ff72a1d99254e97a0ed5a85
.. _eeba077: https://github.com/starccato/starccato_lvk/commit/eeba077d594a001d1ad22393db00729e6697701d


.. _changelog-v0.0.2:

v0.0.2 (2025-10-06)
===================

Bug Fixes
---------

* fix: pypi error (`ebf36f5`_)

Chores
------

* chore(release): 0.0.2 (`a2887d2`_)

Unknown
-------

* add data downloader (`cd4eb0e`_)

* Merge branch 'main' of github.com:starccato/starccato_lvk (`740d1bd`_)

.. _ebf36f5: https://github.com/starccato/starccato_lvk/commit/ebf36f55556f10e3df2e5e8fc465f6cb41458de2
.. _a2887d2: https://github.com/starccato/starccato_lvk/commit/a2887d2ece63ab0267125ee1817a53a3a0b25876
.. _cd4eb0e: https://github.com/starccato/starccato_lvk/commit/cd4eb0ec3fa23cd569a2a068a81961aeb3d6dcdc
.. _740d1bd: https://github.com/starccato/starccato_lvk/commit/740d1bd26d06f8031053ebd20a88551bff50092b


.. _changelog-v0.0.1:

v0.0.1 (2025-10-06)
===================

Bug Fixes
---------

* fix: add missing packages to pyproj (`f0e254f`_)

Chores
------

* chore(release): 0.0.1 (`68f7303`_)

Unknown
-------

* add readme to pyproj (`887d746`_)

* add scipy (`1e7866a`_)

* add prior adjustment (`5c5c8e1`_)

* remove old sampler (`37dfaab`_)

* add main runner interface (`97f9ada`_)

* add sampler (`ef11073`_)

* Remove whitened strain (`7528350`_)

* add workign analysis with real data (`bdeab68`_)

* add minimim SNR test (`844e5c2`_)

* passing with correct psd normalisation (`0b0c059`_)

* simplify test (`173b2f7`_)

* Added more testing (`4eebcfc`_)

* Add fixed time_shift, mre diagnostics, plot the interp PSD (`abcded7`_)

* Add fixed ampliitude (`c34c3ba`_)

* Add simulated PSD test (`39ef54d`_)

* add testing for lnl (`3c6cfca`_)

* hacking on PE (`c060500`_)

* add sampler (`aaf4cfe`_)

* hacking on lnl (`dabb9ef`_)

* add new exe (`66e63a8`_)

* fix float to int bug (`0bdf931`_)

* remove invalid help in click (`c0f0154`_)

* remove invalid help in click (`1d6c00f`_)

* get trigger data for blips and noise (`96004b8`_)

* adjust start stop (`c3c6359`_)

* add pyproj commands (`d4da95d`_)

* more hackig on data acquisition (`e8897db`_)

* add notes (`e86679f`_)

* add testing (`f6b96d3`_)

* add notes (`79134c1`_)

* hacking on lnl (`31f0ce0`_)

* Initial commit (`6974955`_)

.. _f0e254f: https://github.com/starccato/starccato_lvk/commit/f0e254f87b6aef4a9c0cf757475854da40fb134e
.. _68f7303: https://github.com/starccato/starccato_lvk/commit/68f7303fcdbfa352d87894634e79782ea4cf1dcb
.. _887d746: https://github.com/starccato/starccato_lvk/commit/887d7467702e22660d918c14a11925d4b1edc274
.. _1e7866a: https://github.com/starccato/starccato_lvk/commit/1e7866af19444f5b95458b1b17cc3436372d3aef
.. _5c5c8e1: https://github.com/starccato/starccato_lvk/commit/5c5c8e1dc18dfafcdc8c761f23700fee3daa9377
.. _37dfaab: https://github.com/starccato/starccato_lvk/commit/37dfaab5cd6b932205d8925aff74b1fcc02bf625
.. _97f9ada: https://github.com/starccato/starccato_lvk/commit/97f9ada52621c10bb8bc80beb0bd01fe58af4364
.. _ef11073: https://github.com/starccato/starccato_lvk/commit/ef11073835b6731d08c9643f1d102ef7a6c09ac4
.. _7528350: https://github.com/starccato/starccato_lvk/commit/75283506f03b567042ac21971e3d3f33b1c1f6bd
.. _bdeab68: https://github.com/starccato/starccato_lvk/commit/bdeab68b68fb7fca51417ec6758fdb5fceaa4306
.. _844e5c2: https://github.com/starccato/starccato_lvk/commit/844e5c259ffd4a7e7ab4483ccef904ed0b21e457
.. _0b0c059: https://github.com/starccato/starccato_lvk/commit/0b0c05938fe160c60abe708f687cd5e9fdb64cc8
.. _173b2f7: https://github.com/starccato/starccato_lvk/commit/173b2f7c10bbe58c3ae90189d5e4c77003ffdb9b
.. _4eebcfc: https://github.com/starccato/starccato_lvk/commit/4eebcfc5a2c1ad8c0cd81e133aa2b72fbd2acc3a
.. _abcded7: https://github.com/starccato/starccato_lvk/commit/abcded72d4a6b3dab06d640d816cab26baf943bb
.. _c34c3ba: https://github.com/starccato/starccato_lvk/commit/c34c3ba3f0148a0bce260af6264abea5f5c70fba
.. _39ef54d: https://github.com/starccato/starccato_lvk/commit/39ef54d3eec58618be222bc72b42cc1d8842cc8e
.. _3c6cfca: https://github.com/starccato/starccato_lvk/commit/3c6cfcaccace8ba14475a077153d563ee072dd63
.. _c060500: https://github.com/starccato/starccato_lvk/commit/c0605003b23036a49e6af4278172d4c67a9d2130
.. _aaf4cfe: https://github.com/starccato/starccato_lvk/commit/aaf4cfe6cdf85f2ef73c5a5e8ae3d3101e76a3c6
.. _dabb9ef: https://github.com/starccato/starccato_lvk/commit/dabb9ef85d3f2dc89fff6b0c96763098e3c3633b
.. _66e63a8: https://github.com/starccato/starccato_lvk/commit/66e63a843820848be13c1653b6b456dd78c9951c
.. _0bdf931: https://github.com/starccato/starccato_lvk/commit/0bdf931059116f9520ff79c89989ca9e210cdad3
.. _c0f0154: https://github.com/starccato/starccato_lvk/commit/c0f0154495b45bd884029f7b4a8c26bc9f55b7bc
.. _1d6c00f: https://github.com/starccato/starccato_lvk/commit/1d6c00fef2812df1f2cdbfba5477e44d45942bad
.. _96004b8: https://github.com/starccato/starccato_lvk/commit/96004b8901f9f75a4260b817627c71c312ad3acd
.. _c3c6359: https://github.com/starccato/starccato_lvk/commit/c3c635970a5905e6f979e55b5b34e38b159535ac
.. _d4da95d: https://github.com/starccato/starccato_lvk/commit/d4da95db06f8825bf22eac1382704c52fead0883
.. _e8897db: https://github.com/starccato/starccato_lvk/commit/e8897dbec1368e0e9a58994bfbfd893cebd7cd27
.. _e86679f: https://github.com/starccato/starccato_lvk/commit/e86679fd3ac4dee355752c664c6040634efe9e88
.. _f6b96d3: https://github.com/starccato/starccato_lvk/commit/f6b96d35c09b5b3a5686a7a9b740fa02605ee73b
.. _79134c1: https://github.com/starccato/starccato_lvk/commit/79134c1e65562943e1bb0a8dd83e8a8cb3d494d2
.. _31f0ce0: https://github.com/starccato/starccato_lvk/commit/31f0ce015a2fa91d2875b58fffcc33ce21df211c
.. _6974955: https://github.com/starccato/starccato_lvk/commit/6974955e110d1ed275909841fd45f24ed814cfa3
