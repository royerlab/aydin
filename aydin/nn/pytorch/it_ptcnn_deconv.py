import numpy
import torch
import torch.nn.functional as F
from scipy.signal import convolve2d

from aydin.nn.pytorch.it_ptcnn import PTCNNImageTranslator
from aydin.nn.pytorch.models.psf_convolution import PSFConvolutionLayer
from aydin.util.log.log import lprint


def to_numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


class PTCNNDeconvolution(PTCNNImageTranslator):
    """
    Pytorch-based CNN image deconvolution
    """

    def __init__(self, psf_kernel=None, broaden_psf=1, **kwargs):
        """
        Constructs a CNN image translator using the pytorch deep learning library.

        :param normaliser_type: normaliser type
        :param balance_training_data: balance data ? (limits number training entries per target value histogram bin)
        :param monitor: monitor to track progress of training externally (used by UI)
        """
        super().__init__(**kwargs)

        for i in range(broaden_psf):
            psf_kernel = numpy.pad(psf_kernel, (1,), mode='constant', constant_values=0)
            psf_kernel = convolve2d(
                psf_kernel,
                numpy.array([[0.5, 1, 0.5], [1, 1, 1], [0.5, 1, 0.5]]) / 7,
                'same',
            )

        psf_kernel /= psf_kernel.sum()
        psf_kernel = psf_kernel.astype(numpy.float32)

        self.psf_kernel = psf_kernel
        self.psf_kernel_tensor = torch.from_numpy(
            self.psf_kernel[numpy.newaxis, numpy.newaxis, ...]
        ).to(self.device)

        self.psfconv = PSFConvolutionLayer(self.psf_kernel).to(self.device)

        self.enforce_blind_spot = False

        self.sharpening = 0.01

    def _train_loop(self, data_loader, optimizer, loss_function):
        try:
            self.model.kernel_continuity_regularisation = False
        except AttributeError:
            lprint("Cannot deactivate kernel continuity regularisation")

        super()._train_loop(data_loader, optimizer, loss_function)

    def _additional_losses(self, translated_image, forward_model_image):

        # non-negativity loss:
        non_negativity_loss = F.relu(-translated_image)
        loss = non_negativity_loss.mean()

        # Sharpen loss_deconvolution:
        if self.sharpening and self.sharpening > 0:
            sum_values = torch.sum(translated_image, (2, 3), keepdim=True)
            entropy = self.sharpening * torch.sqrt(1 + translated_image / sum_values)
            entropy_loss_value = entropy.mean()
            loss += entropy_loss_value

        return loss

    def _forward_model(self, input):
        return self.psfconv(input)
