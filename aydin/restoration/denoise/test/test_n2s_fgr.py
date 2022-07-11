from pprint import pprint

from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR


def test_configure():

    implementations = Noise2SelfFGR().implementations
    pprint(implementations)

    configurable_arguments = Noise2SelfFGR().configurable_arguments
    pprint(configurable_arguments)

    implementations_description = Noise2SelfFGR().implementations_description
    pprint(implementations_description)
