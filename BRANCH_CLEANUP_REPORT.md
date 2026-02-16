# Branch Cleanup Report

**Date:** 2026-02-15
**Repository:** aydin (royerloic/aydin fork + royerlab/aydin upstream)
**Base branch:** upstream/master at `255458c5`

---

## Summary

Analyzed all remote branches across `origin` (royerloic/aydin) and `upstream` (royerlab/aydin).

**Initial analysis was flawed:** A buggy bash script produced garbled output, causing `merge-base --is-ancestor` results to be misread. All branches were incorrectly reported as "fully merged." On re-verification, only 1 of 8 non-master branches was actually merged. Branches with substantive unmerged content were deleted and then **immediately restored** once the error was discovered. Deleting the `docs-prod` branches also **took down the GitHub Pages documentation site** — this was restored and Pages re-enabled.

**Net result:** Only 1 branch (`max_num_estimators`) was safely deleted. All others were restored.

---

## Branches Safely Deleted (1)

### 1. `origin/max_num_estimators` — DELETED (genuinely merged)
- **Last commit:** `3f8d821b` — "added an override for usage of CPU pinned memory" (2022-08-25)
- **Author:** Loic A. Royer
- **Status:** Truly an ancestor of upstream/master. Zero unique commits, zero unique content.
- **Verdict:** Safe to delete. Nothing lost.

### ~~2. `origin/docs-prod`~~ — RESTORED (GitHub Pages serving branch!)
- **Last commit:** `70015e31` — "docs updated" (2022-03-08)
- **Author:** acs-ws
- **Content:** Generated Sphinx HTML documentation. **This is the serving branch for GitHub Pages on the fork.**
- **Status:** RESTORED after deletion broke the site. **DO NOT DELETE.**

### ~~3. `upstream/docs-prod`~~ — RESTORED (GitHub Pages serving branch!)
- **Last commit:** `876f3852` — "docs updated" (2022-10-29)
- **Author:** AhmetCanSolak
- **Content:** Generated multi-version Sphinx HTML documentation. **This is the serving branch for GitHub Pages at https://royerlab.github.io/aydin/.**
- **Status:** RESTORED and GitHub Pages re-enabled after deletion took down the docs site. **DO NOT DELETE.**

---

## Branches Restored After Erroneous Deletion (4)

### 4. `origin/fgr-cb-new-params` — RESTORED
- **Remote commit:** `bbe06c1a` — "Merge branch 'master' into fgr-cb-new-params" (2022-07-18)
- **Local branch also exists at:** `9d961214` (behind remote)
- **Author:** Loic Royer
- **Unique commits:** 8 (including merge commits)
- **35 files changed, 223 insertions, 56 deletions**

#### Unmerged content NOT in master:

| File | Change | In Master? |
|------|--------|-----------|
| `aydin/features/groups/particle.py` | **New file** — ParticleFeatures class for denoising diffraction-limited molecules. Generates Gaussian kernel features at various sigmas. | **NO** |
| `aydin/features/groups/test/test_particle_feature_group.py` | **New file** — Test for ParticleFeatures | **NO** |
| `aydin/features/standard_features.py` | Adds `include_particle_features`, `particle_min_sigma`, `particle_max_sigma`, `particle_num_filters` parameters | **NO** |
| `aydin/regression/cb.py` | Adds `quantisation_mode` parameter exposing CatBoost's `feature_border_type` (median/uniform/mixed/greedylogsum). Also documents `max_bin` default. | **NO** |
| `aydin/regression/cb.py` | Changes learning rate reduction from `*= 0.5` to `*= 0.25` (more aggressive) | **NO** |
| `aydin/it/transforms/range.py` | Removes final `.astype(self._original_dtype)` cast — type casting should not happen outside normalisation code | **NO** |
| Various files | `lprint` → import renames (already done differently in master) | Already in master (different approach) |

### 5. `origin/latest_fixes` — RESTORED
- **Remote commit:** `0acd8a2a` — "black and flake8 fixes" (2022-05-26)
- **Author:** AhmetCanSolak
- **Unique commits:** 7 (including merge commits)
- **5 files changed, 101 insertions, 7 deletions**

#### Content status — all changes independently applied to master:

| File | Change | In Master? |
|------|--------|-----------|
| `aydin/it/classic_denoisers/butterworth.py` | Adds `t-z-yx` / `xy-z-t` mode for 3D+t timelapse Butterworth denoising (57 lines) | **YES** — already in master |
| `aydin/it/classic_denoisers/spectral.py` | Changes `try_fft` default from `True` to `False`; comments out `@jit` on FFT distance function | **YES** — `try_fft=False` in master |
| `aydin/gui/main_page.py` | Adds try/except with traceback for failed sample image downloads | **YES** — already in master |
| `aydin/util/crop/super_fast_rep_crop.py` | Expands docstring with full parameter documentation | **YES** — already in master |
| `README.md` | Minor additions | Superseded by master's README |

**Verdict:** This branch's content was independently applied to master (likely cherry-picked or re-implemented). The branch itself was never merged, but all its changes are present. **Safe to delete** — restored out of caution but can be re-deleted.

### 6. `origin/m1_friendly` — RESTORED
- **Remote commit:** `5eb975b3` — "more resilience, UI boots now" (2022-05-02)
- **Author:** Loic A. Royer
- **Unique commits:** 4 (including merge commits)
- **10 files changed, 214 insertions, 62 deletions**

#### Unmerged content NOT in master:

| File | Change | In Master? |
|------|--------|-----------|
| `aydin/gui/_qt/custom_widgets/system_summary.py` | Major refactor (140 lines) — wraps `psutil` calls in try/except, adds helper methods (`_cpu_freq()`, `_number_of_cores()`, `_cpu_load_values()`, `_gpu_info()`), makes GUI resilient when psutil/numba unavailable | **NO** |
| `aydin/util/offcore/offcore.py` | Wraps `psutil` import in try/except so offcore array allocation works without psutil | **NO** |
| `aydin/gui/tabs/qt/denoise.py` | Minor resilience fix | **NO** |
| `aydin/it/base.py` | Minor resilience fix | **NO** |
| `aydin/it/fgr.py` | Minor resilience fix | **NO** |
| `aydin/restoration/denoise/noise2selfcnn.py` | Minor resilience fix | **NO** |
| `aydin/restoration/denoise/noise2selffgr.py` | Minor resilience fix | **NO** |
| `aydin/restoration/denoise/util/denoise_utils.py` | Resilience improvements | **NO** |
| `env_osx_m1.sh` | **New file** — M1 conda environment setup script | **NO** |
| `setup.cfg` | Dependency adjustments for M1 | Superseded (migrated to pyproject.toml) |

### 7. `upstream/fix/apple-silicon-compatibility` — RESTORED
- **Remote commit:** `b6a9bb99` — "Fix Apple Silicon (ARM64) compatibility issues" (2025-09-04)
- **Author:** Ilan
- **Unique commits:** 1
- **3 files changed, 61 insertions, 29 deletions**

#### Unmerged content NOT in master:

| File | Change | In Master? |
|------|--------|-----------|
| `aydin/nn/tf/models/utils/conv_block.py` | Wraps `tensorflow.python.keras` imports in try/except falling back to `tensorflow.keras` — fixes import errors on Apple Silicon | **NO** |
| `aydin/regression/nn_utils/models.py` | Same TF Keras import compatibility fix | **NO** |
| `setup.cfg` | Dependency adjustments for ARM64 | Superseded (migrated to pyproject.toml) |

---

## Branches Kept (not deleted) (3)

### 8. `origin/CLNN_KEEP`
- **Commit:** `dd3bda5b` — "saving CLNN for posterity" (2022-04-26)
- **Content:** Custom neural network framework (CLNN) with NumPy and OpenCL backends — 47 files including tensor abstractions, modules (dense, ReLU, losses), optimizers (Adam, SGD), and OpenCL GPU kernels.
- **Status:** Code extracted to external folder per user request. Branch kept as historical bookmark.

### 9. `origin/cnn_pytorch_DONOTDELETE`
- **Commit:** `ce8b1696` — "Merge branch 'master' into cnn_pytorch" (2020-03-25)
- **Content:** Original PyTorch CNN implementation — 415 files, massive diff. Historical significance as the origin of `ImageTranslatorCNNTorch`.
- **Status:** Kept per branch name convention.

### 10. `origin/fix/critical-issues-review`
- **Current working branch** with recent modernization work. Fully merged into upstream/master.

---

## Recommendations

### Worth Considering for Integration

#### 1. `fgr-cb-new-params` — Particle Features & CB Quantisation (HIGH VALUE)
- **ParticleFeatures** (`particle.py`) is a genuinely new feature group for denoising diffraction-limited molecules — useful for microscopy users imaging single molecules
- **CatBoost `quantisation_mode`** exposes an important tuning parameter that's currently hardcoded to `UniformAndQuantiles`
- **Range transform fix** (removing redundant `.astype()`) addresses a real bug where type casting happens outside normalisation
- **Recommendation:** Cherry-pick the particle features, CB quantisation mode, and range.py fix into master. The more aggressive LR reduction (`0.25` vs `0.5`) should be tested before adopting.

#### 2. `m1_friendly` — Resilience & Apple Silicon (MEDIUM VALUE)
- The `psutil` resilience work in `system_summary.py` and `offcore.py` is well-designed — graceful degradation when dependencies are missing
- However, much of it touches files that have changed significantly since 2022 (Qt5→Qt6, setuptools→hatch, etc.), so it would need manual adaptation rather than a clean merge
- `env_osx_m1.sh` is somewhat outdated but the concept is valid
- **Recommendation:** Adapt the psutil-resilience pattern to current codebase. The env script is probably obsolete given modern conda/pip ARM64 support.

#### 3. `upstream/fix/apple-silicon-compatibility` — TF Keras Imports (LOW VALUE)
- The try/except import pattern for `tensorflow.python.keras` vs `tensorflow.keras` addresses a real issue on newer TF versions
- However, TF/Keras CNN is already deprecated in this project in favor of PyTorch (`ImageTranslatorCNNTorch`)
- `setup.cfg` changes are superseded by pyproject.toml migration
- **Recommendation:** Low priority. Only worth it if TF-based CNN support is still maintained. Otherwise, the deprecation of TF CNN makes this moot.

#### 4. `latest_fixes` — All Content Already in Master (NO ACTION NEEDED)
- All substantive changes are already in master
- **Recommendation:** Can be safely deleted. It was restored only out of caution.

### Cleanup Actions Still Available

1. **Delete `origin/latest_fixes`** — content confirmed in master, branch is redundant
2. **Delete stale local branches** — `fgr-cb-new-params` (local, behind remote) and `latest_fixes` (local)
3. **Consider deleting `origin/cnn_pytorch_DONOTDELETE`** — fully merged, historical value only, 6 years old

---

## Lessons Learned

1. **Never trust batch shell scripts with complex output parsing** — the initial garbled output caused a cascade of errors
2. **Always verify merge status with targeted checks** — `merge-base --is-ancestor` should be followed by `git log master..branch` and content-level verification
3. **Check both local AND remote branch positions** — they can diverge when the local branch hasn't been updated
4. **Identify infrastructure branches before deleting** — `docs-prod` was not a stale feature branch but the GitHub Pages serving branch. Deleting it took down the live documentation site. Always check `gh api repos/OWNER/REPO/pages` before deleting any branch that could be serving Pages.
