"""Feature group implementations for extensible feature generation.

Each feature group encapsulates a family of related features that share the
same computation strategy. Feature groups are composed inside an
``ExtensibleFeatureGenerator`` to build the complete feature set used for
self-supervised denoising.

Available feature groups
------------------------
- ``UniformFeatures`` -- multi-scale box (integral) filter features.
- ``SpatialFeatures`` -- spatial coordinate features.
- ``MedianFeatures`` -- median filter features at multiple radii.
- ``LowPassFeatures`` -- Butterworth low-pass filter features.
- ``DCTFeatures`` -- Discrete Cosine Transform basis features.
- ``RandomFeatures`` -- deterministic random convolutional features.
- ``CorrelationFeatures`` -- generic kernel-based correlation features.
- ``LearnedCorrelationFeatures`` -- data-driven convolutional features
  learned via MiniBatchKMeans clustering.
- ``TranslationFeatures`` -- shifted copies of the image as features.
"""
