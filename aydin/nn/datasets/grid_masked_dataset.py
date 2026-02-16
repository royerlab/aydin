"""Dataset with grid-based pixel masking for self-supervised denoising."""

import numpy
import torch
from scipy.ndimage import median_filter
from torch.utils.data import Dataset


class GridMaskedDataset(Dataset):
    """PyTorch Dataset implementing grid-based masking for self-supervised training.

    Generates four masking patterns based on a regular grid, where masked
    pixels are replaced according to the chosen replacement strategy. This
    implements a structured blind-spot strategy for Noise2Self-style training.

    Parameters
    ----------
    image : numpy.ndarray
        Input image tensor with shape ``(B, C, H, W)`` for 2D or
        ``(B, C, D, H, W)`` for 3D images.
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
        replacement_strategy: str = 'interpolate',
    ):
        """Initialize the grid-masked dataset.

        Parameters
        ----------
        image : numpy.ndarray
            Input image tensor with shape ``(B, C, H, W)`` or
            ``(B, C, D, H, W)``.
        replacement_strategy : str
            Replacement strategy for masked pixels.
        """

        self.image = torch.as_tensor(image, dtype=torch.float32)
        self.replacement_strategy = replacement_strategy
        self._image_numpy = image  # Keep numpy copy for median filter

        # Pre-compute all 4 grid masks (only 4 possible phases)
        self._masks = [self._build_mask(i) for i in range(4)]

        # Pre-compute interpolation kernel (avoids recreating on every call)
        ndim = len(image.shape[2:])
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
        """Return the number of grid mask phases (always 4)."""
        return 4

    def _build_mask(self, i):
        """Build a grid mask for the given phase index.

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

        row_offset = phase // 2
        col_offset = phase % 2

        A = torch.zeros(shape)

        if len(self.image.shape) == 4:
            A[:, :, row_offset::2, col_offset::2] = 1
        elif len(self.image.shape) == 5:
            A[:, :, :, row_offset::2, col_offset::2] = 1

        return A

    def get_mask(self, i):
        """Return the cached grid mask for the given phase index.

        Parameters
        ----------
        i : int
            Phase index (0-3) determining the grid offset.

        Returns
        -------
        torch.Tensor
            Binary mask tensor with the same shape as the image.
        """
        return self._masks[i % 4]

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
            # Replace masked pixels with random values from image distribution
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
        """Replace masked pixels with donut-shaped median filter values.

        Parameters
        ----------
        tensor : torch.Tensor
            Input image tensor.
        mask : torch.Tensor
            Binary mask.
        mask_inv : torch.Tensor
            Inverse of the mask.

        Returns
        -------
        torch.Tensor
            Image with masked pixels replaced by local median.
        """
        ndim = len(self.image.shape[2:])
        img_np = tensor.cpu().numpy()

        # Build donut-shaped footprint (exclude center pixel)
        spatial_size = [3] * ndim
        footprint_shape = [1, 1] + spatial_size  # B, C, *spatial
        footprint = numpy.ones(footprint_shape)
        center_idx = tuple([0, 0] + [1] * ndim)
        footprint[center_idx] = 0

        filtered = median_filter(img_np, footprint=footprint)
        filtered_tensor = torch.as_tensor(
            filtered.copy(), device=tensor.device, dtype=tensor.dtype
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

        input_patch = self._apply_replacement(original_patch, mask, mask_inv)

        return original_patch[0], input_patch[0], mask[0]
