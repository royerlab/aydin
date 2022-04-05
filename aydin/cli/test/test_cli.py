from click.testing import CliRunner


from aydin.cli.cli import cli


def test_info():
    runner = CliRunner()
    result = runner.invoke(cli, ['info'])
    assert result.exit_code == 0


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



