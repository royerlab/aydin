import numpy
import torch
from torch.utils.data import Dataset


class GridMaskedDataset(Dataset):
    def __init__(
        self,
        image,
    ):
        """

        Parameters
        ----------
        image
        """

        self.image = torch.tensor(image)

    def __len__(self):
        return 4

    def get_mask(self, i):
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
        original_patch = self.image
        mask = self.get_mask(index)
        mask_inv = torch.ones(mask.shape) - mask

        # input_patch = original_patch * mask_inv
        input_patch = self.interpolate_mask(original_patch, mask, mask_inv)

        return original_patch[0], input_patch[0], mask[0]
