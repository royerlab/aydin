import torch
from torch.utils.data import Dataset


class NoisyGroundtruthDataset(Dataset):
    def __init__(
        self,
        noisy_images,
        groundtruth_images,
        device,
    ):
        """Torch Dataset to handle pairs of noisy and groundtruth images.

        Parameters
        ----------
        noisy_images : numpy.ArrayLike
        groundtruth_images : numpy.ArrayLike

        """

        self.noisy_images = [torch.tensor(image) for image in noisy_images]
        self.groundtruth_images = [torch.tensor(image) for image in groundtruth_images]
        self.device = device

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, index):
        return self.noisy_images[index][0], self.groundtruth_images[index][0]
