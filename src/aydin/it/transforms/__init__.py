"""Image transforms for the denoising pipeline.

Provides a collection of preprocessing/postprocessing transforms that can be
applied before and after image denoising to improve results. Each transform
implements the `ImageTransformBase` interface from `base.py`.

Available transforms:

- `AttenuationTransform` -- corrects axis-aligned intensity attenuation
- `DeskewTransform` -- applies integral shear deskewing for skewed stacks
- `FixedPatternTransform` -- suppresses axis-aligned fixed offset patterns
- `HighpassTransform` -- removes low-frequency content before denoising
- `HistogramEqualisationTransform` -- histogram equalisation / CLAHE
- `MotionStabilisationTransform` -- phase-correlation-based motion correction
- `PaddingTransform` -- pads borders to reduce edge artifacts
- `PeriodicNoiseSuppressionTransform` -- suppresses periodic noise via FFT
- `RangeTransform` -- normalises pixel values to [0, 1]
- `SaltPepperTransform` -- corrects impulse (salt-and-pepper) noise
- `VarianceStabilisationTransform` -- variance stabilisation (Anscombe, etc.)
"""
