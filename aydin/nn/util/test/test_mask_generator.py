import numpy
from aydin.nn.util.mask_generator import maskedgen, randmaskgen


def test_checkerbox_maskgen():
    image = numpy.ones((20, 64, 64, 1))
    batch_size = 10
    mask_size = (3, 3)

    maskgen = maskedgen(image, batch_size, mask_size, replace_by='zero')

    minibatch = next(maskgen)
    actual_mask_ratio = minibatch[0]['input_msk'].sum() / minibatch[0]['input_msk'].size

    assert len(minibatch) == 2
    assert isinstance(minibatch[0], dict)
    assert (
        minibatch[0]['input'].sum()
        == minibatch[0]['input_msk'].size - minibatch[0]['input_msk'].sum()
    )
    assert minibatch[0]['input_msk'].sum() == minibatch[1].sum()
    assert minibatch[1].shape == (batch_size,) + image.shape[1:]
    assert (
        (1 / numpy.prod(mask_size)) * 0.9
        <= actual_mask_ratio
        <= (1 / numpy.prod(mask_size)) * 1.1
    )


def test_random_maskgen():
    image = numpy.ones((20, 64, 64, 1))
    batch_size = 10
    p_maskedpixels = 0.1

    maskgen = randmaskgen(
        image, batch_size, p_maskedpixels=p_maskedpixels, replace_by='zero'
    )

    minibatch = next(maskgen)
    actual_mask_ratio = minibatch[0]['input_msk'].sum() / minibatch[0]['input_msk'].size

    assert len(minibatch) == 2
    assert isinstance(minibatch[0], dict)
    assert (
        minibatch[0]['input'].sum()
        == minibatch[0]['input_msk'].size - minibatch[0]['input_msk'].sum()
    )
    assert minibatch[0]['input_msk'].sum() == minibatch[1].sum()
    assert minibatch[1].shape == (batch_size,) + image.shape[1:]
    assert p_maskedpixels * 0.9 <= actual_mask_ratio <= p_maskedpixels * 1.1
