import torch
from torch.utils.data import Dataset

from aydin.nn.util.random_sample_patches import random_sample_patches


class N2SDataset(Dataset):
    def __init__(
            self,
            image,
            patch_size,
            nb_patches_per_image: int = 8,
            adoption_rate: float = 0.5
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
        )

    def __len__(self):
        return len(self.crop_slicers)

    def get_mask(self, i):
        phase = i % self.grid_size
        shape = self.image[self.crop_slicers[0]].shape
        patch_size = 4

        A = torch.zeros(self.image.shape[1:-1])

        if len(self.image.shape) == 4:
            for i in range(shape[-3]):
                for j in range(shape[-2]):
                    if i % patch_size == phase and j % patch_size == phase:
                        A[i, j] = 1

        elif len(self.image.shape) == 5:
            for i in range(shape[-4]):
                for j in range(shape[-3]):
                    for k in range(shape[-2]):
                        if i % patch_size == phase and j % patch_size == phase and k % patch_size == phase:
                            A[i, j, k] = 1

        return torch.Tensor(A)

    def __getitem__(self, index):
        original_patch = self.image[self.crop_slicers[index]]
        mask = self.get_mask(0)
        mask_inv = torch.ones(mask.shape).to(original_patch.device) - mask

        input_patch = original_patch * mask_inv

        return original_patch, input_patch, mask
