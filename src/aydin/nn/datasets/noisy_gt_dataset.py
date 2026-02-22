"""Dataset for paired noisy and ground-truth images."""

import torch
from torch.utils.data import Dataset


class NoisyGroundtruthDataset(Dataset):
    """PyTorch Dataset for paired noisy and ground-truth images.

    Converts numpy arrays to torch tensors and provides indexed access
    to noisy/ground-truth image pairs for supervised training.

    Parameters
    ----------
    noisy_images : list of numpy.ndarray
        List of noisy input images.
    groundtruth_images : list of numpy.ndarray
        List of corresponding clean ground-truth images.
    device : torch.device
        Device to store the tensors on.
    """

    def __init__(
        self,
        noisy_images,
        groundtruth_images,
        device,
    ):
        """Initialize the noisy/ground-truth dataset.

        Parameters
        ----------
        noisy_images : list of numpy.ndarray
            List of noisy input images.
        groundtruth_images : list of numpy.ndarray
            List of corresponding clean ground-truth images.
        device : torch.device
            Device to store the tensors on.
        """

        self.noisy_images = [
            torch.as_tensor(image, dtype=torch.float32) for image in noisy_images
        ]
        self.groundtruth_images = [
            torch.as_tensor(image, dtype=torch.float32) for image in groundtruth_images
        ]
        self.device = device

    def __len__(self):
        """Return the number of image pairs in the dataset."""
        return len(self.noisy_images)

    def __getitem__(self, index):
        """Return the noisy and ground-truth image pair at the given index.

        Parameters
        ----------
        index : int
            Index of the image pair.

        Returns
        -------
        tuple of torch.Tensor
            Tuple of (noisy_image, groundtruth_image) tensors.
        """
        return self.noisy_images[index][0], self.groundtruth_images[index][0]
