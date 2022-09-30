import numpy
import torch
from torch.utils.data import Dataset


class RandomMaskedDataset(Dataset):
    def __init__(
        self,
        image,
        nb_masks=4,
        pixel_masking_probability: float = 0.3,
    ):
        """

        Parameters
        ----------
        image
        """

        self.image = torch.tensor(image)
        self.nb_masks = nb_masks
        self.p = pixel_masking_probability

    def __len__(self):
        return self.nb_masks

    def get_mask(self):
        shape = self.image.shape

        mask = torch.rand(shape)
        mask[mask > self.p] = 1
        mask[mask <= self.p] = 0
        mask = mask.bool()

        return mask

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
        mask = self.get_mask()
        mask_inv = torch.ones(mask.shape) - mask

        # input_patch = original_patch * mask_inv
        input_patch = self.interpolate_mask(original_patch, mask, mask_inv)

        return original_patch[0], input_patch[0], mask[0]
