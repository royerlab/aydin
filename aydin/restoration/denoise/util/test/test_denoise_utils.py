from pprint import pprint

from aydin.restoration.denoise.util.denoise_utils import (
    get_list_of_denoiser_implementations,
)


def est_denoise_utils():
    implementations = get_list_of_denoiser_implementations()
    pprint(implementations)
