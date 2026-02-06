"""Demo script for discovering available Noise2Self FGR implementations."""

from pprint import pprint

from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR


def demo_fgr_discovery():
    """Print all available FGR implementations, their arguments, and descriptions."""

    implementations = Noise2SelfFGR().implementations
    pprint(implementations)

    configurable_arguments = Noise2SelfFGR().configurable_arguments
    pprint(configurable_arguments)

    implementations_description = Noise2SelfFGR().implementations_description
    pprint(implementations_description)


if __name__ == '__main__':
    demo_fgr_discovery()
