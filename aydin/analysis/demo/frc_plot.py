# flake8: noqa
# import pytest
import pytest
import matplotlib.pyplot as plt
from numpy.random.mtrand import normal

from aydin.analysis.fsc import fsc
from aydin.io import imread
from aydin.io.datasets import camera, normalise
from aydin.util.log.log import lprint, Log


if __name__ == "__main__":
    denoised = imread("~/Dev/AhmetCanSolak/aydin/denoised_live-1.png")[0]
    denoised = normalise(denoised)[:357, :357]

    image = imread("~/Dev/AhmetCanSolak/aydin/raw_live.png")[0]
    raw_image = normalise(image)[:357, :357]

    print(denoised.shape, raw_image.shape)

    noise1 = normal(size=raw_image.size).reshape(*raw_image.shape)

    morenoisy_image = raw_image + noise1

    corr_raw_morenoisy = fsc(raw_image, morenoisy_image)
    corr_raw_denoised = fsc(raw_image, denoised)
    corr_morenoisy_denoised = fsc(morenoisy_image, denoised)

    plt.plot(corr_raw_morenoisy, "-b", label="FRC raw/more_noisy")

    plt.legend(loc="upper right")
    # plt.title('raw, more_noisy')

    plt.savefig("single-frc-lines.png")
    plt.show()

    plt.plot(corr_raw_morenoisy, "-b", label="FRC raw/more_noisy")
    plt.plot(corr_raw_denoised, "-r", label="FRC raw/denoised")
    plt.plot(corr_morenoisy_denoised, "-g", label="FRC more_noisy/denoised")
    plt.legend(loc="upper right")

    plt.savefig("tri-frc-lines.png")
    plt.show()
