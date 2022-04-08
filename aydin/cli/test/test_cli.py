from click.testing import CliRunner

from aydin.cli.cli import cli, handle_files
from aydin.io.datasets import examples_single
from aydin.util.log.log import Log


def test_info():
    Log.override_test_exclusion = True
    Log.force_click_echo = True

    image_path = examples_single.generic_lizard.get_path()

    runner = CliRunner()
    result = runner.invoke(cli, ['info', image_path])

    assert result.exit_code == 0
    assert "Reading" in result.output
    assert "Metadata" in result.output
    assert "batch" in result.output


def test_cite():
    runner = CliRunner()
    result = runner.invoke(cli, ['cite'])
    print(result.output)
    assert result.exit_code == 0
    assert "10.5281/zenodo.5654826" in result.output


def test_handle_files():
    file_list = [
        examples_single.generic_lizard.get_path(),
        examples_single.fountain.get_path()
    ]
    filepaths, image_arrays, metadatas = handle_files(file_list, "")

    assert filepaths == file_list

    assert image_arrays[0].shape == examples_single.generic_lizard.get_array().shape
    assert image_arrays[0].dtype == examples_single.generic_lizard.get_array().dtype
    assert image_arrays[1].shape == examples_single.fountain.get_array().shape
    assert image_arrays[1].dtype == examples_single.fountain.get_array().dtype

    assert metadatas[0].shape == examples_single.generic_lizard.get_array().shape
    assert metadatas[0].dtype == examples_single.generic_lizard.get_array().dtype
    assert metadatas[1].shape == examples_single.fountain.get_array().shape
    assert metadatas[1].dtype == examples_single.fountain.get_array().dtype


def test_denoise():
    runner = CliRunner()
    result = runner.invoke(cli, ['denoise'])
    assert result.exit_code == 1
