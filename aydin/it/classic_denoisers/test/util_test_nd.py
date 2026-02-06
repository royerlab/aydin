import numpy


def check_nd(denoiser):
    """Test denoiser works on 1D-4D inputs with realistic data."""
    numpy.random.seed(42)
    shape = (17, 7, 13, 9)

    for ndim in range(1, 5):
        image = numpy.random.rand(*shape[:ndim]).astype(numpy.float32)
        result = denoiser(image)
        assert result is not None
        assert result.shape == image.shape
        # Verify denoiser actually processed the image (not just returning input)
        assert not numpy.array_equal(result, image)
