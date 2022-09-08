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
        self.image = image

        self.crop_slicers = random_sample_patches(
            image,
            patch_size=patch_size,
            nb_patches_per_image=nb_patches_per_image,
            adoption_rate=adoption_rate,
        )

    def __len__(self):
        return len(self.crop_slicers)

    def __getitem__(self, index):
        input_image = self.image[self.crop_slicers[index]]

        return input_image
