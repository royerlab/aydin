import torch
from torch.utils.data import Dataset

from aydin.nn.models.torch.demo.unet.original_n2s import interpolate_mask
from aydin.nn.util.random_sample_patches import random_sample_patches


class N2SDataset(Dataset):
    def __init__(
        self,
        image,
        patch_size,
        nb_patches_per_image: int = 64,
        adoption_rate: float = 0.2,
    ):
        """

        Parameters
        ----------
        image
        patch_size
        nb_patches_per_image
        adoption_rate
        """

        self.image = torch.tensor(image)

        self.crop_slicers = random_sample_patches(
            image,
            patch_size=patch_size,
            nb_patches_per_image=nb_patches_per_image,
            adoption_rate=adoption_rate,
            backend="torch",
        )

    def __len__(self):
        return 4  # len(self.crop_slicers)

    def get_mask(self, i):
        phase = i % 4
        # shape = self.image[self.crop_slicers[1]].shape
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

    def __getitem__(self, index):
        # original_patch = self.image[self.crop_slicers[index]]
        original_patch = self.image
        mask = self.get_mask(index)
        mask_inv = torch.ones(mask.shape) - mask

        # input_patch = original_patch * mask_inv
        input_patch = interpolate_mask(original_patch, mask, mask_inv)

        return original_patch[0], input_patch[0], mask[0]
