from aydin.io.datasets import examples_single


def demo_noise_floor_delta():
    noisy_image = examples_single.gauss_noisy.get_array()

    noise_floor_delta = noise_floor_delta_over_z(denoised_image, noisy_image)


demo_noise_floor_delta()
