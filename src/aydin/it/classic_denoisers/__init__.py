"""Classic image denoisers with auto-calibration via J-invariance.

This package provides a collection of classical denoising algorithms, each
paired with an auto-calibration function that finds optimal parameters using
the Noise2Self (N2S) J-invariance loss. Available denoisers include:

- **bilateral** -- Edge-preserving bilateral filter
- **bmnd** -- Block-Matching nD (generalized BM3D)
- **butterworth** -- Butterworth low-pass filter
- **dictionary_fixed** -- Sparse coding over a fixed DCT/DST dictionary
- **dictionary_learned** -- Sparse coding over a learned dictionary
- **gaussian** -- Gaussian low-pass filter
- **gm** -- Gaussian-Median mix filter
- **harmonic** -- Non-linear harmonic prior
- **lipschitz** -- Lipschitz continuity enforcement (impulse noise removal)
- **nlm** -- Non-Local Means
- **pca** -- PCA-based patch denoising
- **spectral** -- Spectral (DCT/DST/FFT) patch thresholding
- **tv** -- Total Variation (Bregman and Chambolle)
- **wavelet** -- Wavelet thresholding (BayesShrink, VisuShrink)

Each module exposes two main functions:

- ``calibrate_denoise_<name>(image, ...)`` -- Finds optimal parameters and
  returns ``(denoise_function, best_parameters, memory_needed)``.
- ``denoise_<name>(image, ...)`` -- Applies the denoiser with given parameters.
"""
