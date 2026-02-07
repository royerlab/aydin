"""Dataset with random pixel masking for self-supervised denoising."""

import numpy
import torch
from torch.utils.data import Dataset

from aydin.nn.datasets.random_patches import random_patches


class RandomMaskedDataset(Dataset):
    """PyTorch Dataset implementing random pixel masking for self-supervised training.

    Extracts random patches from the input image and applies random pixel
    masking where masked pixels are replaced by interpolated neighbor values.
    Implements a Noise2Self-style blind-spot strategy with random masks.

    Parameters
    ----------
    image : numpy.ndarray
        Input image tensor with shape ``(B, C, ...spatial_dims...)``.
    patch_size : int
        Spatial size of extracted patches.
    pixel_masking_probability : float
        Probability of each pixel being masked during training.
    """

    def __init__(
        self,
        image,
        patch_size: int = 32,
        pixel_masking_probability: float = 0.3,
    ):
        """Initialize the random-masked dataset.

        Parameters
        ----------
        image : numpy.ndarray
            Input image tensor with shape ``(B, C, ...spatial_dims...)``.
        patch_size : int
            Spatial size of extracted patches.
        pixel_masking_probability : float
            Probability of each pixel being masked.
        """

        self.image = torch.tensor(image)
        self.patch_size = patch_size

        # print("shape: ", image.shape)

        self.patch_slicing_objects = random_patches(
            image=image,
            patch_size=patch_size,
            nb_patches_per_image=min(
                numpy.prod(image.shape) // numpy.prod(patch_size), 1040
            ),
        )
        self.p = pixel_masking_probability

    def __len__(self):
        """Return the number of patches in the dataset."""
        return len(self.patch_slicing_objects)

    def get_mask(self):
        """Generate a random binary mask for a single patch.

        Returns
        -------
        torch.Tensor
            Binary mask tensor with shape ``(1, 1, ...spatial_dims...)``,
            where 1 indicates blind-spot (masked) pixels and 0 indicates context pixels.
        """
        shape = (self.patch_size,) * len(self.image.shape[2:])

        mask = (torch.rand(shape) <= self.p).float()
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)

        return mask

    def interpolate_mask(self, tensor, mask, mask_inv):
        """Replace masked pixels with interpolated neighbor values.

        Parameters
        ----------
        tensor : torch.Tensor
            Input image patch tensor.
        mask : torch.Tensor
            Binary mask where 1 indicates pixels to replace.
        mask_inv : torch.Tensor
            Inverse of the mask (1 - mask).

        Returns
        -------
        torch.Tensor
            Patch with masked pixels replaced by neighbor-weighted averages.
        """
        device = tensor.device

        mask = mask.to(device)
        mask_inv = mask_inv.to(device)

        ndim = len(self.image.shape[2:])

        if ndim == 2:
            kernel = numpy.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
            kernel = kernel[numpy.newaxis, numpy.newaxis, :, :]
            kernel = torch.Tensor(kernel).to(device)
            kernel = kernel / kernel.sum()
            filtered_tensor = torch.nn.functional.conv2d(
                tensor, kernel, stride=1, padding=1
            )
        elif ndim == 3:
            kernel = numpy.array(
                [
                    [[0.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 0.0]],
                    [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                    [[0.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 0.0]],
                ]
            )
            kernel = kernel[numpy.newaxis, numpy.newaxis, :, :, :]
            kernel = torch.Tensor(kernel).to(device)
            kernel = kernel / kernel.sum()
            filtered_tensor = torch.nn.functional.conv3d(
                tensor, kernel, stride=1, padding=1
            )
        else:
            raise ValueError(f"Unsupported number of spatial dimensions: {ndim}")

        return filtered_tensor * mask + tensor * mask_inv

    def __getitem__(self, index):
        """Return the original, masked-input, and mask tensors for a patch.

        Parameters
        ----------
        index : int
            Patch index.

        Returns
        -------
        tuple of torch.Tensor
            Tuple of (original_patch, input_patch, mask) tensors.
        """
        original_patch = self.image[self.patch_slicing_objects[index]]
        mask = self.get_mask()
        mask_inv = torch.ones(mask.shape) - mask

        # input_patch = original_patch * mask_inv
        input_patch = self.interpolate_mask(original_patch, mask, mask_inv)

        # print(original_patch.shape, input_patch.shape, mask.shape)

        return original_patch[0], input_patch[0], mask[0]
