import numpy
import torch
import torch.nn.functional as F


def richardson_lucy_pytorch(image, psf, iterations=50, clip=True, donut=False):
    use_cuda = True
    device_index = 0
    device = torch.device(f"cuda:{device_index}" if use_cuda else "cpu")
    # print(f"Using device: {device}")

    image = image.astype(numpy.float)
    psf = psf.astype(numpy.float)
    im_deconv = numpy.full(image.shape, image.mean())
    psf_mirror = psf[::-1, ::-1].copy()
    psf_size = psf_mirror.shape[0]

    image = (
        torch.from_numpy(image[numpy.newaxis, numpy.newaxis, ...]).float().to(device)
    )
    psf = torch.from_numpy(psf[numpy.newaxis, numpy.newaxis, ...]).float().to(device)
    psf_mirror = (
        torch.from_numpy(psf_mirror[numpy.newaxis, numpy.newaxis, ...])
        .float()
        .to(device)
    )
    im_deconv = (
        torch.from_numpy(im_deconv[numpy.newaxis, numpy.newaxis, ...])
        .float()
        .to(device)
    )

    for _ in range(iterations):
        pad = (psf_size - 1) // 2
        convolved = F.conv2d(F.pad(im_deconv, (pad, pad, pad, pad), "reflect"), psf)
        relative_blur = image / convolved

        im_deconv *= F.conv2d(
            F.pad(relative_blur, (pad, pad, pad, pad), "reflect"), psf_mirror
        )

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < 0] = 0

    return im_deconv.detach().cpu().numpy().squeeze()
