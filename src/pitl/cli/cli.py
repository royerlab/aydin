import os
import numpy as np
import click
from skimage.data import camera

from src.pitl.services.Noise2Self import Noise2Self
from src.pitl.util.resource import read_image_from_path
from ..examples.demo_pitl_2D_cli import demo_pitl_2D

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='0.0.1')
def pitl():
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
