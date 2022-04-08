from click.testing import CliRunner

from aydin.cli.cli import cli
from aydin.io.datasets import examples_single
from aydin.util.log.log import Log


def test_info():
    Log.override_test_exclusion = True
    image_path = examples_single.generic_lizard.get_path()

    runner = CliRunner()
    result = runner.invoke(cli, ['info', image_path])
    print(result.output)
    print(result.stdout_bytes)

    assert result.exit_code == 0
    assert "Reading" in result.stdout_bytes
    print(result.output)
    assert "Metadata" in result.stdout


def test_cite():
    runner = CliRunner()
    result = runner.invoke(cli, ['cite'])
    print(result.output)
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
