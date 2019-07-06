import os
import click
import logging

from src.pitl.gui import gui
from src.pitl.services.Noise2Self import Noise2Self
from src.pitl.util.resource import read_image_from_path
from ..examples.demo_pitl_2D_cli import demo_pitl_2D
import sentry_sdk


logger = logging.getLogger(__name__)


def absPath(myPath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    import sys

    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        logger.debug("found MEIPASS: %s "%os.path.join(base_path, os.path.basename(myPath)))

        return os.path.join(base_path, os.path.basename(myPath))
    except Exception as e:
        logger.debug("did not find MEIPASS: %s "%e)

        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.version_option(version='0.0.1')
def pitl(ctx):
    sentry_sdk.init("https://d9d7db5f152546c490995a409023c60a@sentry.io/1498298")
    if ctx.invoked_subcommand is None:
        gui.run()
    else:
        pass


@pitl.command()
@click.argument('mode')
def demo(**kwargs):
    if kwargs['mode'] == '2D':
        print("Running demo_pitl_2D")
        demo_pitl_2D()
    else:
        print("Rest of the demos not support by cli yet, sorry :(")


@pitl.command()
@click.argument('path')
def noise2self(**kwargs):
    path = os.path.abspath(kwargs['path'])
    noisy = read_image_from_path(path)
    Noise2Self.run(noisy)


if __name__ == '__main__':
    pitl()
