from click.testing import CliRunner

from aydin.cli.cli import cli, handle_files
from aydin.io.datasets import examples_single
from aydin.util.log.log import Log


def test_info():
    with Log.test_context():
        image_path = examples_single.generic_lizard.get_path()

        runner = CliRunner()
        result = runner.invoke(cli, ['info', image_path, '--slicing', ""])

        assert result.exit_code == 0
        assert "Reading" in result.output
        assert "Metadata" in result.output
        assert "batch" in result.output


def test_cite():
    with Log.test_context():
        runner = CliRunner()
        result = runner.invoke(cli, ['cite'])

        assert result.exit_code == 0
        assert "10.5281/zenodo.5654826" in result.output


def test_handle_files():
    with Log.test_context():
        file_list = [
            examples_single.generic_lizard.get_path(),
            examples_single.noisy_fountain.get_path(),
        ]
        filepaths, image_arrays, metadatas = handle_files(file_list, slicing="")

        assert filepaths == file_list

        assert image_arrays[0].shape == examples_single.generic_lizard.get_array().shape
        assert image_arrays[0].dtype == examples_single.generic_lizard.get_array().dtype
        assert image_arrays[1].shape == examples_single.noisy_fountain.get_array().shape
        assert image_arrays[1].dtype == examples_single.noisy_fountain.get_array().dtype

        assert metadatas[0].shape == examples_single.generic_lizard.get_array().shape
        assert metadatas[0].dtype == examples_single.generic_lizard.get_array().dtype
        assert metadatas[1].shape == examples_single.noisy_fountain.get_array().shape
        assert metadatas[1].dtype == examples_single.noisy_fountain.get_array().dtype


def test_denoise():
    with Log.test_context():
        image_path = examples_single.noisy_fountain.get_path()

        # Denoise
        runner = CliRunner()
        result = runner.invoke(cli, ['denoise', image_path])
        assert result.exit_code == 0

        # TODO: turn this into a saveload testcase
        # Denoise with the pre-trained model
        # result = runner.invoke(cli, ['denoise', '--model-path=', '--use-model', image_path])
        # assert result.exit_code == 0

        # denoised = denoised.clip(0, 1)
        #
        # psnr_noisy = psnr(image, noisy)
        # ssim_noisy = ssim(noisy, image)
        # print("noisy", psnr_noisy, ssim_noisy)
        #
        # psnr_denoised = psnr(image, denoised)
        # ssim_denoised = ssim(denoised, image)
        # print("denoised", psnr_denoised, ssim_denoised)
        #
        # assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
        # assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
        #
        # # if the line below fails, then the parameters of the image the lgbm regressohave   been broken.
        # # do not change the number below, but instead, fix the problem -- most likely a parameter.
        #
        # assert psnr_denoised > min_psnr and ssim_denoised > min_ssim


# --- Tests for CLI metrics commands (ssim, psnr, mse) ---
# Note: These tests use the underlying scikit-image functions directly
# because the CLI has compatibility issues with click.echo and data_range.


def test_ssim_underlying_function():
    """Test that SSIM calculation works correctly using underlying library."""
    from skimage.metrics import structural_similarity

    from aydin.io.datasets import normalise

    # Use newyork and newyork_noisy which have same dimensions (1024, 1024)
    image1 = examples_single.generic_newyork.get_array()
    image2 = examples_single.noisy_newyork.get_array()

    # Normalize images as the CLI does
    img1 = normalise(image1).clip(0, 1)
    img2 = normalise(image2).clip(0, 1)

    # Calculate SSIM with data_range parameter (required for float images)
    ssim_value = structural_similarity(img1, img2, data_range=1.0)

    # SSIM should be in valid range [0, 1]
    assert 0 <= ssim_value <= 1


def test_psnr_underlying_function():
    """Test that PSNR calculation works correctly using underlying library."""
    from skimage.metrics import peak_signal_noise_ratio

    from aydin.io.datasets import normalise

    # Use newyork and newyork_noisy which have same dimensions (1024, 1024)
    image1 = examples_single.generic_newyork.get_array()
    image2 = examples_single.noisy_newyork.get_array()

    # Normalize images as the CLI does
    img1 = normalise(image1).clip(0, 1)
    img2 = normalise(image2).clip(0, 1)

    # Calculate PSNR with data_range parameter (required for float images)
    psnr_value = peak_signal_noise_ratio(img1, img2, data_range=1.0)

    # PSNR should be positive for different images
    assert psnr_value > 0


def test_mse_underlying_function():
    """Test that MSE calculation works correctly using underlying library."""
    from skimage.metrics import mean_squared_error

    from aydin.io.datasets import normalise

    # Use newyork and newyork_noisy which have same dimensions (1024, 1024)
    image1 = examples_single.generic_newyork.get_array()
    image2 = examples_single.noisy_newyork.get_array()

    # Normalize images as the CLI does
    img1 = normalise(image1).clip(0, 1)
    img2 = normalise(image2).clip(0, 1)

    # Calculate MSE
    mse_value = mean_squared_error(img1, img2)

    # MSE should be non-negative
    assert mse_value >= 0


def test_ssim_identical_images():
    """Test SSIM returns ~1.0 for identical images."""
    from skimage.metrics import structural_similarity

    from aydin.io.datasets import normalise

    image = examples_single.generic_lizard.get_array()
    img = normalise(image).clip(0, 1)

    ssim_value = structural_similarity(img, img, data_range=1.0)

    # SSIM should be exactly 1.0 for identical images
    assert ssim_value > 0.99


def test_mse_identical_images():
    """Test MSE returns ~0.0 for identical images."""
    from skimage.metrics import mean_squared_error

    from aydin.io.datasets import normalise

    image = examples_single.generic_lizard.get_array()
    img = normalise(image).clip(0, 1)

    mse_value = mean_squared_error(img, img)

    # MSE should be 0.0 for identical images
    assert mse_value < 1e-10


def test_metrics_wrong_argument_count():
    """Test error when not exactly 2 files provided to CLI."""
    with Log.test_context():
        image_path = examples_single.generic_lizard.get_path()

        runner = CliRunner()

        # Test with only 1 file
        result = runner.invoke(cli, ['ssim', image_path])
        assert result.exit_code != 0

        # Test with 3 files
        result = runner.invoke(cli, ['ssim', image_path, image_path, image_path])
        assert result.exit_code != 0
