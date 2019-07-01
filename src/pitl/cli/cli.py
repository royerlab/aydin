import click

from ..examples.demo_pitl_2D_camera import demo_pitl_2D

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


if __name__ == '__main__':
    pitl()
