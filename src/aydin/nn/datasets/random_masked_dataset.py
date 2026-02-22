"""Dataset with random pixel masking for self-supervised denoising."""

import numpy
import torch
from scipy.ndimage import median_filter
from torch.utils.data import Dataset

from aydin.nn.datasets.random_patches import random_patches


class RandomMaskedDataset(Dataset):
    """PyTorch Dataset implementing random pixel masking for self-supervised training.

    Extracts random patches from the input image and applies random pixel
    masking where masked pixels are replaced according to the chosen
    replacement strategy. Implements a Noise2Self-style blind-spot strategy
    with random masks.

    Parameters
    ----------
    image : numpy.ndarray
        Input image tensor with shape ``(B, C, ...spatial_dims...)``.
    patch_size : int
        Spatial size of extracted patches.
    pixel_masking_probability : float
        Probability of each pixel being masked during training.
    replacement_strategy : str
        Strategy for replacing masked pixels:
        - ``'interpolate'`` (default): weighted neighbor average
        - ``'zero'``: replace with zeros
        - ``'random'``: replace with random values from the image distribution
        - ``'median'``: replace with donut-shaped median filter
    """

    def __init__(
        self,
        image,
        patch_size: int = 32,
        pixel_masking_probability: float = 0.3,
        replacement_strategy: str = 'interpolate',
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
        replacement_strategy : str
            Replacement strategy for masked pixels.
        """

        self.image = torch.as_tensor(image, dtype=torch.float32)
        self.patch_size = patch_size
        self.replacement_strategy = replacement_strategy

        # Validate patch_size against spatial dimensions
        spatial_shape = image.shape[2:]
        min_spatial = min(spatial_shape)
        if patch_size > min_spatial:
            raise ValueError(
                f"patch_size ({patch_size}) exceeds smallest spatial "
                f"dimension ({min_spatial}) of image shape {image.shape}"
            )

        # Use spatial dims only for patch count; raise patch_size to ndim power
        ndim = len(spatial_shape)
        nb_patches = max(
            1,
            min(int(numpy.prod(spatial_shape) // (patch_size**ndim)), 1040),
        )

        self.patch_slicing_objects = random_patches(
            image=image,
            patch_size=patch_size,
            nb_patches_per_image=nb_patches,
        )
        self.p = pixel_masking_probability

        # Pre-compute interpolation kernel (avoids recreating on every call)
        if ndim == 2:
            kernel = numpy.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
            kernel = kernel[numpy.newaxis, numpy.newaxis, :, :]
        elif ndim == 3:
            kernel = numpy.array(
                [
                    [[0.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 0.0]],
                    [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                    [[0.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 0.0]],
                ]
            )
            kernel = kernel[numpy.newaxis, numpy.newaxis, :, :, :]
        else:
            kernel = None
        if kernel is not None:
            kernel = kernel / kernel.sum()
        self._interp_kernel = (
            torch.as_tensor(kernel, dtype=torch.float32) if kernel is not None else None
        )
        self._spatial_ndim = ndim

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

        num_channels = tensor.shape[1]
        kernel = self._interp_kernel.to(device)

        if self._spatial_ndim == 2:
            kernel = kernel.expand(num_channels, 1, -1, -1)
            filtered_tensor = torch.nn.functional.conv2d(
                tensor, kernel, stride=1, padding=1, groups=num_channels
            )
        elif self._spatial_ndim == 3:
            kernel = kernel.expand(num_channels, 1, -1, -1, -1)
            filtered_tensor = torch.nn.functional.conv3d(
                tensor, kernel, stride=1, padding=1, groups=num_channels
            )
        else:
            raise ValueError(
                f"Unsupported number of spatial dimensions: {self._spatial_ndim}"
            )

        return filtered_tensor * mask + tensor * mask_inv

    def _apply_replacement(self, tensor, mask, mask_inv):
        """Apply the configured replacement strategy to masked pixels.

        Parameters
        ----------
        tensor : torch.Tensor
            Input image tensor.
        mask : torch.Tensor
            Binary mask (1 = pixel to replace).
        mask_inv : torch.Tensor
            Inverse of the mask.

        Returns
        -------
        torch.Tensor
            Image with masked pixels replaced.
        """
        if self.replacement_strategy == 'interpolate':
            return self.interpolate_mask(tensor, mask, mask_inv)
        elif self.replacement_strategy == 'zero':
            return tensor * mask_inv
        elif self.replacement_strategy == 'random':
            img_min = tensor.min().item()
            img_max = tensor.max().item()
            random_values = torch.rand_like(tensor) * (img_max - img_min) + img_min
            return tensor * mask_inv + random_values * mask
        elif self.replacement_strategy == 'median':
            return self._median_replacement(tensor, mask, mask_inv)
        else:
            raise ValueError(
                f"Unknown replacement_strategy: {self.replacement_strategy}"
            )

    def _median_replacement(self, tensor, mask, mask_inv):
        """Replace masked pixels with donut-shaped median filter values."""
        ndim = len(self.image.shape[2:])
        img_np = tensor.cpu().numpy()

        spatial_size = [3] * ndim
        footprint_shape = [1, 1] + spatial_size
        footprint = numpy.ones(footprint_shape)
        center_idx = tuple([0, 0] + [1] * ndim)
        footprint[center_idx] = 0

        filtered = median_filter(img_np, footprint=footprint)
        filtered_tensor = torch.as_tensor(
            filtered.copy(), device=tensor.device, dtype=tensor.dtype
        )

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

        input_patch = self._apply_replacement(original_patch, mask, mask_inv)

        return original_patch[0], input_patch[0], mask[0]
