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
        # psnr_noisy = psnr(noisy, image)
        # ssim_noisy = ssim(noisy, image)
        # print("noisy", psnr_noisy, ssim_noisy)
        #
        # psnr_denoised = psnr(denoised, image)
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
