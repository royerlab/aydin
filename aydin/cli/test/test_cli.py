from click.testing import CliRunner

from aydin.cli.cli import cli
from aydin.io.datasets import examples_single


def test_info():
    image_path = examples_single.generic_lizard.get_path()
    print(image_path)

    runner = CliRunner()
    result = runner.invoke(cli, ['info', image_path])
    print(result.output)
    assert result.exit_code == 0
    assert "batch" in result.output


def test_cite():
    runner = CliRunner()
    result = runner.invoke(cli, ['cite'])
    assert result.exit_code == 0
    assert "10.5281/zenodo.5654826" in result.output


def test_handle_files():
    runner = CliRunner()
    result = runner.invoke(cli, ['handle_files'])
    assert result.exit_code == 0


def test_denoise():
    runner = CliRunner()
    result = runner.invoke(cli, ['denoise'])
    assert result.exit_code == 1
