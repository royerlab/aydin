"""Dataset with grid-based pixel masking for self-supervised denoising."""

import numpy
import torch
from torch.utils.data import Dataset


class GridMaskedDataset(Dataset):
    """PyTorch Dataset implementing grid-based masking for self-supervised training.

    Generates four masking patterns based on a regular grid, where masked
    pixels are replaced by interpolated values from their neighbors. This
    implements a structured blind-spot strategy for Noise2Self-style training.

    Parameters
    ----------
    image : numpy.ndarray
        Input image tensor with shape ``(B, C, H, W)`` for 2D or
        ``(B, C, D, H, W)`` for 3D images.
    """

    def __init__(
        self,
        image,
    ):
        """Initialize the grid-masked dataset.

        Parameters
        ----------
        image : numpy.ndarray
            Input image tensor with shape ``(B, C, H, W)`` or
            ``(B, C, D, H, W)``.
        """

        self.image = torch.tensor(image)

    def __len__(self):
        """Return the number of grid mask phases (always 4)."""
        return 4

    def get_mask(self, i):
        """Generate a grid mask for the given phase index.

        Parameters
        ----------
        i : int
            Phase index (0-3) determining the grid offset.

        Returns
        -------
        torch.Tensor
            Binary mask tensor with the same shape as the image.
        """
        phase = i % 4
        shape = self.image.shape
        patch_size = 4

        A = torch.zeros(shape)

        if len(self.image.shape) == 4:
            for i in range(shape[-2]):
                for j in range(shape[-1]):
                    if i % patch_size == phase and j % patch_size == phase:
                        A[:, :, i, j] = 1

        elif len(self.image.shape) == 5:
            for i in range(shape[-3]):
                for j in range(shape[-2]):
                    for k in range(shape[-1]):
                        if (
                            i % patch_size == phase
                            and j % patch_size == phase
                            and k % patch_size == phase
                        ):
                            A[:, :, i, j, k] = 1

        return torch.Tensor(A)

    def interpolate_mask(self, tensor, mask, mask_inv):
        """Replace masked pixels with interpolated neighbor values.

        Parameters
        ----------
        tensor : torch.Tensor
            Input image tensor.
        mask : torch.Tensor
            Binary mask where 1 indicates pixels to replace.
        mask_inv : torch.Tensor
            Inverse of the mask (1 - mask).

        Returns
        -------
        torch.Tensor
            Image with masked pixels replaced by neighbor-weighted averages.
        """
        device = tensor.device

        mask = mask.to(device)
        mask_inv = mask_inv.to(device)

        kernel = numpy.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
        kernel = kernel[numpy.newaxis, numpy.newaxis, :, :]
        kernel = torch.Tensor(kernel).to(device)
        kernel = kernel / kernel.sum()

        filtered_tensor = torch.nn.functional.conv2d(
            tensor, kernel, stride=1, padding=1
        )

        return filtered_tensor * mask + tensor * mask_inv

    def __getitem__(self, index):
        """Return the original, masked-input, and mask tensors for a phase.

        Parameters
        ----------
        index : int
            Phase index (0-3).

        Returns
        -------
        tuple of torch.Tensor
            Tuple of (original_patch, input_patch, mask) tensors.
        """
        original_patch = self.image
        mask = self.get_mask(index)
        mask_inv = torch.ones(mask.shape) - mask

        # input_patch = original_patch * mask_inv
        input_patch = self.interpolate_mask(original_patch, mask, mask_inv)

        return original_patch[0], input_patch[0], mask[0]
