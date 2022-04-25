from aydin.analysis.noise_floor_delta import noise_floor_delta
from aydin.io.datasets import examples_single


def demo_noise_floor_delta():
    noisy_image = examples_single.gauss_noisy.get_array()

    noise_floor_delta = noise_floor_delta(denoised_image, noisy_image, axis=0)


demo_noise_floor_delta()
