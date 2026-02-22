"""Feature generation module for self-supervised image denoising.

This module provides feature generators that transform input images into
multi-dimensional feature arrays used by regression models for self-supervised
denoising. The main components are:

- ``FeatureGeneratorBase`` -- abstract base class defining the feature
  generator interface (serialization, compute, receptive field).
- ``ExtensibleFeatureGenerator`` -- composable feature generator that
  assembles features from multiple ``FeatureGroupBase`` instances.
- ``StandardFeatureGenerator`` -- pre-configured generator combining
  uniform, spatial, median, DCT, low-pass, and random convolutional
  features for the FGR denoising approach.

Feature groups (in the ``groups`` subpackage) encapsulate individual
families of related features such as multi-scale box filters, median
filters, Butterworth low-pass filters, and spatial coordinate features.
"""
