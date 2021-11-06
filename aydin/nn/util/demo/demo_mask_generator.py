# flake8: noqa
import numpy as np
from aydin.it.cnn.util.mask_generator import masker, maskedgen, randmaskgen


def demo_maskgen():  # TODO: refactor this demo into a test
    batch_vol = (64, 64)
    mask_shape = (3, 3)
    batch_size = 8
    image = np.ones((64, 64, 64, 1))

    masked_chkr = maskedgen(batch_vol, mask_shape, image, batch_size)
    masked_rndm = randmaskgen(image, batch_size, p_maskedpixels=0.1)

    for m_rndm in masked_rndm:
        break
    for m_chkr in masked_chkr:
        break

    a = masker(batch_vol, 0, mask_shape, 0.1)
    b = masker(batch_vol, 0, mask_shape, None)
