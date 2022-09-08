from torch.utils.data import Dataset

from aydin.nn.util.random_sample_patches import random_sample_patches


class N2SDataset(Dataset):
    def __init__(self, image, patch_size):
        """ """
        self.image = image

        self.crop_slicers = random_sample_patches(
            image,
            patch_size=patch_size
        )

    def __len__(self):
        return len(self.crop_slicers)

    def __getitem__(self, index):
        input = self.image[self.crop_slicers[index]]

        return input
