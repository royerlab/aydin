import numpy
import torch
import torch.nn.functional as F
from torch import nn


def lucy_richardson_loss(observed_image, candidate_image, psf_kernel, mask=None):
    if mask is not None:
        candidate_image = candidate_image * mask

    padding = list((x - 1) // 2 for x in psf_kernel.shape)[-2:]
    convolved = F.conv2d(candidate_image, psf_kernel, padding=padding)
    kernel_psf_mirror = torch.flip(psf_kernel, (1, 2))
    loss = candidate_image * (
        F.conv2d(observed_image / convolved, kernel_psf_mirror, padding=padding) - 1
    )

    loss = loss.abs() * mask

    return loss


class LucyRichardson(nn.Module):
    def __init__(
        self, psf_kernel, num_channels_in=1, num_channels_out=1, iterations=4, clip=True
    ):
        super().__init__()

        self.clip = clip
        self.num_channels_out = num_channels_out
        self.num_channels_in = num_channels_in
        self.iterations = iterations

        self.psf_size = psf_kernel.shape[0]

        psf_kernel = psf_kernel.astype(numpy.float)
        kernel_psf_mirror = psf_kernel[::-1, ::-1].copy()

        self.kernel_psf_tensor = torch.from_numpy(
            psf_kernel[numpy.newaxis, numpy.newaxis, ...]
        ).float()
        self.kernel_psf_mirror_tensor = torch.from_numpy(
            kernel_psf_mirror[numpy.newaxis, numpy.newaxis, ...]
        ).float()

        self.kernel_psf_tensor = torch.nn.Parameter(self.kernel_psf_tensor)
        self.kernel_psf_mirror_tensor = torch.nn.Parameter(
            self.kernel_psf_mirror_tensor
        )

    def forward(self, x):

        im_deconv = 0.5 * torch.ones_like(x)

        for _ in range(self.iterations):
            convolved = F.conv2d(
                im_deconv, self.kernel_psf_tensor, padding=(self.psf_size - 1) // 2
            )
            relative_blur = x / convolved
            im_deconv = im_deconv * F.conv2d(
                relative_blur,
                self.kernel_psf_mirror_tensor,
                padding=(self.psf_size - 1) // 2,
            )

        if self.clip:
            im_deconv.clamp_(-1, 1)

        return im_deconv

    def post_optimisation(self):
        with torch.no_grad():
            self.kernel_psf_tensor += torch.min(self.kernel_psf_tensor)
            self.kernel_psf_tensor /= torch.sum(self.kernel_psf_tensor)
