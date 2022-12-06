import numpy
import torch
from torch.utils.data import Dataset

from aydin.nn.datasets.random_patches import random_patches


class RandomMaskedDataset(Dataset):
    def __init__(
        self,
        image,
        patch_size: int = 32,
        pixel_masking_probability: float = 0.3,
    ):
        """

        Parameters
        ----------
        image
        """

        self.image = torch.tensor(image)
        self.patch_size = patch_size

        # print("shape: ", image.shape)

        self.patch_slicing_objects = random_patches(
            image=image,
            patch_size=patch_size,
            nb_patches_per_image=min(image.size / numpy.prod(patch_size), 1040),
        )
        self.p = pixel_masking_probability

    def __len__(self):
        return len(self.patch_slicing_objects)

    def get_mask(self):
        shape = (self.patch_size,) * len(self.image.shape[2:])

        mask = torch.rand(shape)
        mask[mask > self.p] = 1
        mask[mask <= self.p] = 0
        # mask = mask.bool()
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)

        return mask

    def interpolate_mask(self, tensor, mask, mask_inv):
        device = tensor.device

        mask = mask.to(device)
        mask_inv = mask_inv.to(device)

        if len(self.image.shape) == 4:
            kernel = numpy.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
        elif len(self.image.shape) == 5:
            kernel = numpy.array([
                [[0.5, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 0.5]],
                [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                [[0.5, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 0.5]],
            ])
        else:
            raise ValueError("Image can only have 2 or 3 spacetime dimensions...")

        kernel = kernel[numpy.newaxis, numpy.newaxis, :, :]
        kernel = torch.Tensor(kernel).to(device)
        kernel = kernel / kernel.sum()

        conv_method = torch.nn.functional.conv2d if len(self.image.shape) == 4 else torch.nn.functional.conv3d

        filtered_tensor = conv_method(
            tensor, kernel, padding=1,  # stride=1,
        )

        return filtered_tensor * mask + tensor * mask_inv

    def __getitem__(self, index):
        original_patch = self.image[self.patch_slicing_objects[index]]
        mask = self.get_mask()
        mask_inv = torch.ones(mask.shape) - mask

        # input_patch = original_patch * mask_inv
        input_patch = self.interpolate_mask(original_patch, mask, mask_inv)

        # print(original_patch.shape, input_patch.shape, mask.shape)

        return original_patch[0], input_patch[0], mask[0]
