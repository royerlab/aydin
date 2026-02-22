"""Demonstrate querying the best available GPU device for CNN inference.

This script instantiates a CNN Torch image translator and prints the name
of the best available compute device (GPU or CPU fallback).
"""

from aydin.it.cnn_torch import ImageTranslatorCNNTorch

it = ImageTranslatorCNNTorch()

print(it.get_best_device_name())
