"""Tests for the Aydin command-line interface."""

import os
import shutil

import click
import numpy
import pytest
from click.testing import CliRunner

from aydin.cli.cli import cli, handle_files
from aydin.io.datasets import examples_single
from aydin.io.io import imread, imwrite
from aydin.util.log.log import Log


def test_info():
    """Test the 'info' CLI command returns image metadata."""
    with Log.test_context():
        image_path = examples_single.generic_lizard.get_path()

        runner = CliRunner()
        result = runner.invoke(cli, ['info', image_path, '--slicing', ""])

        assert result.exit_code == 0
        assert "Reading" in result.output
        assert "Metadata" in result.output
        assert "batch" in result.output


def test_info_with_slicing():
    """Test 'info' CLI command with a slicing argument."""
    with Log.test_context():
        image_path = examples_single.generic_lizard.get_path()

        runner = CliRunner()
        result = runner.invoke(cli, ['info', image_path, '-s', '[0:100,0:100]'])

        assert result.exit_code == 0
        assert "Reading" in result.output


def test_cite():
    """Test the 'cite' CLI command outputs the Zenodo DOI."""
    with Log.test_context():
        runner = CliRunner()
        result = runner.invoke(cli, ['cite'])

        assert result.exit_code == 0
        assert "10.5281/zenodo.5654826" in result.output


def test_version_flag():
    """Test the --version flag prints the version string."""
    from aydin import __version__

    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_main_help_shows_banner():
    """Test that main --help output includes the Aydin ASCII banner."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])

    assert result.exit_code == 0
    # The figlet banner contains this distinctive pattern
    assert '|___/' in result.output
    assert 'Self-supervised image denoising' in result.output


def test_list_denoisers_shows_categories():
    """Test --list-denoisers groups denoisers by category."""
    with Log.test_context():
        runner = CliRunner()
        result = runner.invoke(cli, ['denoise', '--list-denoisers'])

        assert result.exit_code == 0
        assert 'Classic' in result.output
        assert 'Noise2SelfFGR' in result.output


def test_handle_files():
    """Test that handle_files correctly loads images and metadata from file paths."""
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


def test_handle_files_nonexistent():
    """Test that handle_files raises BadParameter for nonexistent files."""
    with Log.test_context():
        with pytest.raises(click.BadParameter, match="file not found"):
            handle_files(['nonexistent_file.tif'], slicing='')


def test_list_denoisers():
    """Test --list-denoisers prints available denoisers without HTML tags."""
    with Log.test_context():
        runner = CliRunner()
        result = runner.invoke(cli, ['denoise', '--list-denoisers'])

        assert result.exit_code == 0
        # Denoiser names are case-insensitive in the listing
        output_lower = result.output.lower()
        assert 'butterworth' in output_lower
        assert 'noise2self' in output_lower or 'fgr' in output_lower
        # HTML tags should be stripped for terminal display
        assert '<a href' not in result.output


def test_denoise():
    """Test the 'denoise' CLI command runs successfully on an example image."""
    with Log.test_context():
        image_path = examples_single.noisy_fountain.get_path()

        runner = CliRunner()
        result = runner.invoke(cli, ['denoise', image_path])
        assert result.exit_code == 0


def test_denoise_with_variant():
    """Test denoise CLI with explicit denoiser variant."""
    with Log.test_context():
        image_path = examples_single.noisy_fountain.get_path()

        runner = CliRunner()
        result = runner.invoke(cli, ['denoise', '-d', 'classic-gaussian', image_path])
        assert result.exit_code == 0


# --- CLI metric command tests ---


def test_ssim(paired_image_files):
    """Test the 'ssim' CLI command produces correct output."""
    with Log.test_context():
        clean_path, noisy_path = paired_image_files

        runner = CliRunner()
        result = runner.invoke(cli, ['ssim', clean_path, noisy_path])

        assert result.exit_code == 0, result.output
        assert "ssim:" in result.output.lower()


def test_psnr(paired_image_files):
    """Test the 'psnr' CLI command produces correct output."""
    with Log.test_context():
        clean_path, noisy_path = paired_image_files

        runner = CliRunner()
        result = runner.invoke(cli, ['psnr', clean_path, noisy_path])

        assert result.exit_code == 0, result.output
        assert "psnr:" in result.output.lower()


def test_mse(paired_image_files):
    """Test the 'mse' CLI command produces correct output."""
    with Log.test_context():
        clean_path, noisy_path = paired_image_files

        runner = CliRunner()
        result = runner.invoke(cli, ['mse', clean_path, noisy_path])

        assert result.exit_code == 0, result.output
        assert "mse:" in result.output.lower()


def test_ssim_identical_images(single_image_file):
    """Test SSIM returns ~1.0 for identical images via CLI."""
    with Log.test_context():
        runner = CliRunner()
        result = runner.invoke(cli, ['ssim', single_image_file, single_image_file])

        assert result.exit_code == 0, result.output
        # Extract numeric value from output line like "├ ssim:  1.0"
        for line in result.output.splitlines():
            if 'ssim:' in line.lower():
                value = float(line.split('ssim:')[1].strip())
                assert (
                    value > 0.99
                ), f"SSIM for identical images should be ~1.0, got {value}"
                break


def test_mse_identical_images(single_image_file):
    """Test MSE returns ~0.0 for identical images via CLI."""
    with Log.test_context():
        runner = CliRunner()
        result = runner.invoke(cli, ['mse', single_image_file, single_image_file])

        assert result.exit_code == 0, result.output
        # Extract numeric value from output line like "├ mse:  0.0"
        for line in result.output.splitlines():
            if 'mse:' in line.lower():
                value = float(line.split('mse:')[1].strip())
                assert (
                    value < 1e-10
                ), f"MSE for identical images should be ~0, got {value}"
                break


def test_metrics_wrong_argument_count():
    """Test error when not exactly 2 files provided to metric CLI commands."""
    with Log.test_context():
        image_path = examples_single.generic_lizard.get_path()

        runner = CliRunner()

        # Test with only 1 file
        result = runner.invoke(cli, ['ssim', image_path])
        assert result.exit_code != 0

        # Test with 3 files
        result = runner.invoke(cli, ['ssim', image_path, image_path, image_path])
        assert result.exit_code != 0


def test_metrics_nonexistent_file():
    """Test metric commands fail with nonexistent files."""
    with Log.test_context():
        runner = CliRunner()
        result = runner.invoke(cli, ['ssim', 'nonexistent1.tif', 'nonexistent2.tif'])
        assert result.exit_code != 0


def test_fsc(paired_image_files):
    """Test the 'fsc' CLI command produces the output plot."""
    with Log.test_context():
        clean_path, noisy_path = paired_image_files

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['fsc', clean_path, noisy_path])

            assert result.exit_code == 0, result.output
            assert os.path.exists("fsc.png")


def test_fsc_custom_output(paired_image_files):
    """Test the 'fsc' CLI command with custom output path."""
    with Log.test_context():
        clean_path, noisy_path = paired_image_files

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli, ['fsc', clean_path, noisy_path, '-o', 'custom_plot.png']
            )

            assert result.exit_code == 0, result.output
            assert os.path.exists("custom_plot.png")
            assert not os.path.exists("fsc.png")


def test_split_channels_output_files(tmp_path):
    """Test split_channels produces correct per-channel output files."""
    with Log.test_context():
        # Copy RGB test image to tmp_path so output filenames are valid
        src_path = examples_single.rgbtest.get_path()
        local_path = str(tmp_path / "rgbtest.png")
        shutil.copy2(src_path, local_path)

        runner = CliRunner()
        result = runner.invoke(cli, ['split-channels', local_path])

        assert result.exit_code == 0, result.output

        # Verify each channel file exists and has the correct shape
        original, _ = imread(local_path)
        num_channels = original.shape[-1]  # RGB image has C as last axis

        for i in range(num_channels):
            channel_path = str(tmp_path / f"channel_{i}_rgbtest.png")
            assert os.path.exists(channel_path), f"Missing channel file: {channel_path}"

            channel_img, _ = imread(channel_path)
            # Each channel should be 2D (the C axis is removed)
            assert (
                channel_img.ndim == 2
            ), f"Channel {i} should be 2D, got shape {channel_img.shape}"


def test_hyperstack_output_shape(tmp_path):
    """Test hyperstack produces an output file with the correct stacked shape."""
    with Log.test_context():
        rng = numpy.random.RandomState(42)
        shape = (32, 32)
        img1 = rng.uniform(0, 1, shape).astype(numpy.float32)
        img2 = rng.uniform(0, 1, shape).astype(numpy.float32)

        path1 = str(tmp_path / "img1.tif")
        path2 = str(tmp_path / "img2.tif")
        imwrite(img1, path1)
        imwrite(img2, path2)

        runner = CliRunner()
        result = runner.invoke(cli, ['hyperstack', path1, path2])

        assert result.exit_code == 0, result.output

        # Find the hyperstacked output file
        output_files = [f for f in os.listdir(tmp_path) if 'hyperstacked' in f]
        assert (
            len(output_files) == 1
        ), f"Expected 1 hyperstacked file, found: {output_files}"

        stacked, _ = imread(str(tmp_path / output_files[0]))
        assert stacked.shape == (
            2,
            32,
            32,
        ), f"Expected shape (2, 32, 32), got {stacked.shape}"


def test_denoise_invalid_model_extension(single_image_file):
    """Test denoise fails when model path has wrong extension."""
    with Log.test_context():
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'denoise',
                '--use-model',
                '--model-path',
                'model.tar',
                single_image_file,
            ],
        )

        assert result.exit_code != 0
        assert "must be a .zip archive" in result.output


def test_denoise_use_model_without_model_path(single_image_file):
    """Test denoise --use-model fails when --model-path is missing."""
    with Log.test_context():
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ['denoise', '--use-model', single_image_file],
        )

        assert result.exit_code != 0
        assert "--model-path is required" in result.output


def test_denoise_help_text():
    """Test that denoise --help shows meaningful option descriptions."""
    runner = CliRunner()
    result = runner.invoke(cli, ['denoise', '--help'])

    assert result.exit_code == 0
    # Check metavar placeholders appear
    assert 'VARIANT' in result.output
    assert 'SLICE' in result.output
    assert 'PATH' in result.output
    # Check help descriptions appear
    assert 'Denoiser variant' in result.output
    assert 'List available denoiser variants' in result.output


# --- Heavy tests (full denoise train cycle) ---


@pytest.mark.heavy
def test_denoise_saves_model(tmp_path):
    """Test denoise with --output-folder saves both denoised image and model."""
    with Log.test_context():
        image_path = examples_single.noisy_fountain.get_path()
        output_folder = str(tmp_path / "output")
        os.makedirs(output_folder)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'denoise',
                '-d',
                'classic-gaussian',
                '--output-folder',
                output_folder,
                image_path,
            ],
        )

        assert result.exit_code == 0, result.output

        output_contents = os.listdir(output_folder)

        # A denoised image file should exist
        denoised_files = [f for f in output_contents if 'denoised' in f]
        assert (
            len(denoised_files) >= 1
        ), f"No denoised image found in {output_folder}: {output_contents}"

        # A model zip file should exist
        model_files = [
            f for f in output_contents if 'model' in f and f.endswith('.zip')
        ]
        assert (
            len(model_files) >= 1
        ), f"No model .zip found in {output_folder}: {output_contents}"


@pytest.mark.heavy
def test_denoise_use_saved_model(tmp_path):
    """Test full save/load roundtrip: train -> save model -> reload and denoise."""
    with Log.test_context():
        image_path = examples_single.noisy_fountain.get_path()
        output_folder = str(tmp_path / "output")
        os.makedirs(output_folder)

        runner = CliRunner()

        # Step 1: Denoise and save model
        result1 = runner.invoke(
            cli,
            [
                'denoise',
                '-d',
                'classic-gaussian',
                '--output-folder',
                output_folder,
                image_path,
            ],
        )
        assert result1.exit_code == 0, result1.output

        # Find the saved model zip
        model_files = [
            f for f in os.listdir(output_folder) if 'model' in f and f.endswith('.zip')
        ]
        assert (
            len(model_files) >= 1
        ), f"No model .zip found after first denoise: {os.listdir(output_folder)}"
        model_path = os.path.join(output_folder, model_files[0])

        # Step 2: Denoise again using saved model
        output_folder2 = str(tmp_path / "output2")
        os.makedirs(output_folder2)

        result2 = runner.invoke(
            cli,
            [
                'denoise',
                '--use-model',
                '--model-path',
                model_path,
                '--output-folder',
                output_folder2,
                image_path,
            ],
        )
        assert result2.exit_code == 0, result2.output

        # Verify denoised output exists from second run
        denoised2_files = [f for f in os.listdir(output_folder2) if 'denoised' in f]
        assert (
            len(denoised2_files) >= 1
        ), f"No denoised image from model reuse: {os.listdir(output_folder2)}"

        # Pixel comparison: saved model should produce numerically equivalent results
        denoised1_files = [f for f in os.listdir(output_folder) if 'denoised' in f]
        denoised1, _ = imread(os.path.join(output_folder, denoised1_files[0]))
        denoised2, _ = imread(os.path.join(output_folder2, denoised2_files[0]))
        numpy.testing.assert_array_almost_equal(denoised1, denoised2, decimal=2)


# --- Startup performance regression test ---


def test_cli_import_time():
    """CLI module should import in under 2 seconds (no eager heavy imports)."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import time; t=time.perf_counter(); '
            'from aydin.cli.cli import cli; '
            'elapsed=time.perf_counter()-t; '
            'assert elapsed < 2.0, f"Import took {elapsed:.1f}s"',
        ],
        capture_output=True,
        timeout=10,
    )
    assert (
        result.returncode == 0
    ), f"CLI import too slow or failed:\n{result.stderr.decode()}"
