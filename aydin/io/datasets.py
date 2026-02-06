"""Example datasets for testing and demonstration.

This module provides convenience functions to download, cache, and load
example images from Google Drive for testing Aydin denoising algorithms.
It also includes utility functions for adding synthetic noise and blur.
"""

import os
import zipfile
from enum import Enum
from os.path import exists, join
from typing import Optional

import gdown
import numpy
from scipy.ndimage import binary_dilation, zoom
from scipy.signal import convolve, convolve2d
from skimage.exposure import rescale_intensity
from skimage.util import random_noise

from aydin.io import io
from aydin.io.folders import get_cache_folder
from aydin.util.log.log import lprint
from aydin.util.psf.simple_microscope_psf import SimpleMicroscopePSF

datasets_folder = join(get_cache_folder(), 'data')

try:
    os.makedirs(datasets_folder)
except Exception:
    pass


def normalise(image):
    """Normalize an image to the [0, 1] range as float32.

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        Input image of any numeric type.

    Returns
    -------
    normalized : numpy.ndarray
        Image rescaled to [0, 1] with dtype float32.
    """
    return rescale_intensity(image, in_range='image', out_range=(0, 1)).astype(
        numpy.float32, copy=False
    )


def add_noise(
    image,
    intensity=5,
    variance=0.01,
    sap=0.0,
    dtype=numpy.float32,
    clip=True,
    seed: Optional[int] = None,
):
    """Add synthetic noise to an image.

    Applies Poisson noise (shot noise), Gaussian noise, and salt-and-pepper
    noise sequentially.

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        Input image, ideally normalized to [0, 1].
    intensity : float or None
        Poisson noise intensity. Higher values mean less noise.
        If None, Poisson noise is skipped.
    variance : float
        Variance of the additive Gaussian noise.
    sap : float
        Amount of salt-and-pepper noise (fraction of pixels affected).
    dtype : numpy.dtype
        Output data type.
    clip : bool
        If True, clips the output to the valid range.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    noisy : numpy.ndarray
        Noisy image with the specified dtype.
    """
    if seed is not None:
        seed = abs(seed)
        numpy.random.seed(seed)
    noisy = image
    if intensity is not None:
        noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode="gaussian", var=variance, rng=seed, clip=clip)
    noisy = random_noise(noisy, mode="s&p", amount=sap, rng=seed, clip=clip)
    noisy = noisy.astype(dtype, copy=False)
    return noisy


def add_blur_2d(image, dz=0):
    """Apply 2D blur to an image using a simulated microscope PSF.

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        2D input image.
    dz : int
        Axial offset index into the 3D PSF to select the 2D slice.

    Returns
    -------
    blurred : numpy.ndarray
        Blurred image with the same shape as ``image``.
    psf_kernel : numpy.ndarray
        The normalized PSF kernel that was applied.
    """
    psf = SimpleMicroscopePSF()
    psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)
    psf_kernel = psf_xyz_array[dz]
    psf_kernel /= psf_kernel.sum()
    return convolve2d(image, psf_kernel, mode='same', boundary="symm"), psf_kernel


def add_blur_3d(image, xy_size=17, z_size=17):
    """Apply 3D blur to an image using a simulated microscope PSF.

    Parameters
    ----------
    image : numpy.typing.ArrayLike
        3D input image.
    xy_size : int
        Lateral size of the PSF kernel in pixels.
    z_size : int
        Axial size of the PSF kernel in pixels.

    Returns
    -------
    blurred : numpy.ndarray
        Blurred image with the same shape as ``image``.
    psf_kernel : numpy.ndarray
        The normalized 3D PSF kernel that was applied.
    """
    psf = SimpleMicroscopePSF()
    psf_xyz_array = psf.generate_xyz_psf(
        dxy=0.406, dz=0.406, xy_size=xy_size, z_size=z_size
    )
    psf_kernel = psf_xyz_array
    psf_kernel /= psf_kernel.sum()
    return convolve(image, psf_kernel, mode='same'), psf_kernel


# Convenience shortcuts:


def lizard():
    """Load the example lizard image (2D monochrome)."""
    return examples_single.generic_lizard.get_array()


def camera():
    """Load the example camera image (2D monochrome)."""
    return examples_single.generic_camera.get_array()


def newyork():
    """Load the example New York image (2D monochrome)."""
    return examples_single.generic_newyork.get_array()


def small_newyork():
    """Load a downscaled (0.5x) version of the New York image."""
    return zoom(newyork(), zoom=0.5)


def cropped_newyork(crop_amount=256):
    """Load a center-cropped version of the New York image."""
    crop_amount = min(crop_amount, 500)
    return newyork()[crop_amount:-crop_amount, crop_amount:-crop_amount]


def newyork_noisy():
    """Load the noisy version of the New York image."""
    return examples_single.noisy_newyork.get_array()


def pollen():
    """Load the example pollen image (2D monochrome)."""
    return examples_single.generic_pollen.get_array()


def scafoldings():
    """Load the example scaffoldings image (2D monochrome)."""
    return examples_single.generic_scafoldings.get_array()


def characters():
    """Load the example characters image (2D monochrome, inverted)."""
    return 1 - examples_single.generic_characters.get_array()


def andromeda():
    """Load the example Andromeda galaxy image (2D monochrome)."""
    return examples_single.generic_andromeda.get_array()


def dots():
    """Generate a synthetic sparse dots image with non-uniform background."""
    image = numpy.random.rand(512, 512) < 0.005  # andromeda()#[256:-256, 256:-256]
    image = 0.8 * binary_dilation(image).astype(numpy.float32, copy=False)
    image[0:256, 0:256] += 0.1
    image.clip(0, 1)
    return image


def rgbtest():
    """Load the example RGB test image."""
    return examples_single.rgbtest.get_array()


def dmel():
    """Load a single slice of the Keller Drosophila melanogaster dataset."""
    return examples_single.keller_dmel.get_array()[23]


class examples_single(Enum):
    """Enumeration of single example images available for download.

    Each member is a tuple of (Google Drive file ID, filename).
    Use ``get_path()`` to download and get the local path, or
    ``get_array()`` to directly get the image as a numpy array.
    """

    def get_path(self):
        """Download (if needed) and return the local file path for this example."""
        download_from_gdrive(*self.value, datasets_folder)
        return join(datasets_folder, self.value[1])

    def get_array(self):
        """Download (if needed), read, and return the image as a numpy array."""
        array, _ = io.imread(self.get_path())
        return array

    # XY natural images (2D monochrome):
    generic_camera = ('1S205p0oI-dEQIFbBuFu3QSlMqWA2xk6B', 'camera.png')
    generic_crowd = ('13UHK8MjhBviv31mAW2isdG4G-aGaNJIj', 'crowd.tif')
    generic_mandrill = ('1B33ELiFuCV0OJ6IHh7Ix9lvImwI_QkR-', 'mandrill.tif')
    generic_newyork = ('15Nuu_NU3iNuoPRmpFbrGIY0VT0iCmuKu', 'newyork.png')
    generic_lizard = ('1GUc6jy5QH5DaiUskCrPrf64YBOLzT6j1', 'lizard.png')
    generic_pollen = ('1S0o2NWtD1shB5DfGRIqOFxTLOi8cHQD-', 'pollen.png')
    generic_scafoldings = ('1ZiWhHnkuaQH-BS8B71y00wkN1Ylo38nY', 'scafoldings.png')
    generic_andromeda = ('1Zl3DtkwUlZSbvpxGILexiIoLW1JOdJh8', 'andromeda.png')

    # XY noisy (2D monochrome):
    noisy_fountain = ('1JP-_j-6U7J1gNc9IZCZ_GsgXTcybmZgS', 'fountain.png')
    noisy_newyork = ('13ompUqT7Ti64fStqx76I9j9voWMZWnfA', 'newyork_noisy.tif')
    noisy_monalisa = ('15T3oTCyz7ugnPLTsKc0a9NT17g9GJsO_', 'monalisa.png')
    noisy_gauss = ('17e_ECJA7DUQGu9JELbTkAKbOVVE9olHN', 'Gauss_noisy.png')
    noisy_brown_chessboard = (
        '1gnqwhZ7HrRaScj6QF_P2Pl_6WAcLzCgR',
        'Brown_SIDD_chessboard_gray.png',
    )

    # Patterned noise (2D monochrome)
    periodic_noise = ('1HfwF6gnzHFFdJ-tozllU_h14vNk9GZOG', 'periodic_noise.png')

    # Characters (2D monochrome, inverted):
    generic_characters = ('1ZWkHFI2iddKa9qv6tft4QZlCoDS5fLMK', 'characters.jpg')

    # XYC (RGB)
    rgbtest = ('1KvhcGBqEQ5N9mwxHwy14NVp8OJ-9GCsH', 'rgbtest.png')

    # Leonetti datasets:
    leonetti_tm7sf2 = (
        '1HHsbZ6jyuJkIj6c7kGtsPKOgpUxo0ihw',
        'Leonetti_p4B3_1_TM7SF2_PyProcessed_IJClean.tif',
    )
    leonetti_sptssa = (
        '10kR7FSIyi7417XYTLrMJaGe3MfvMmSYA',
        'Leonetti_p1H10_2_SPTSSA_PyProcessed_IJClean.tif',
    )
    leonetti_snca = (
        '1UyF5HkZLwTaoiBf1sLHkTdw09yyCJyKO',
        'Leonetti_p1H8_2_SNCA_PyProcessed_IJClean.tif',
    )
    leonetti_arhgap21 = (
        '1arq6nj4oiJaxG7dPHjhYVTSYXM2Czpgx',
        'Leonetti_OC-FOV_ARHGAP21_ENSG00000107863_CID000556_FID00030711_stack.tif',
    )
    leonetti_ankrd11 = (
        '1Bl0WlEPeDe8MmlWy8_KaZS9QKthvSDHM',
        'Leonetti_OC-FOV_ANKRD11_ENSG00000167522_CID001385_FID00033338_stack.tif',
    )

    # XYZ
    huang_fixed_pattern_noise = (
        '1fiVopWwUSZJhsWUfoo7YaU3qQ-j4g5hB',
        'fixed_pattern_noise.tif',
    )

    keller_dmel = (
        '12DCAlDRSiTyGDSD7p06nk17GO3ztHg-Q',
        'SPC0_TM0132_CM0_CM1_CHN00_CHN01.fusedStack.tif',
    )

    janelia_flybrain = (
        '12Z6W_f3TqCsl_okKmaLcVUBgS6xEvdjj',
        'Flybrain_3ch_mediumSize.tif',
    )

    myers_tribolium = (
        '1GGULAF8IoPi4P614Wi-njmeQxND4Xttj',
        'Myers_Tribolium_nGFP_0.1_0.2_0.5_20_13_late.tif',
    )

    royerlab_hcr = (
        '1nkgqs8VkmPKBtBHqUXKrVgIvOqD8B84h',
        'Royer_confocal_dragonfly_hcr_drerio_30somite_crop.tif',
    )

    machado_drosophile_egg_chamber = (
        '1msjf1pVAGsy61QMtxvVoxxk5WZCofdN2',
        'C2-DrosophilaEggChamber-small.tif',
    )

    # 2D+t
    cognet_nanotube1 = (
        '1SmrBheUc6p5qTgtIEzedCwbN87HOW_O_',
        'Cognet_r03-s01-100mW-20ms-175 50xplpeg-173.tif',
    )
    cognet_nanotube_400fps = (
        '1ap4YNxa0RA6MBKiXZ2ZRL1USZtzPqFs3',
        'Cognet_1-400fps.tif',
    )
    cognet_nanotube_200fps = (
        '1Z501FlQOBQmPaeBMCOGy6chBDh1bDjEf',
        'Cognet_1-200fps.tif',
    )
    cognet_nanotube_100fps = (
        '1T4UvbF3MRgT4jO4ExIHprvTqUXLiMjyA',
        'Cognet_1-100fps.tif',
    )

    # 3D+t
    maitre_mouse = (
        '13b0-6PUo2YEWG8Z3M1pVfBQBWE0PtILK',
        'Maitre_mouse blastocyst_fracking_180124_e3_crop.tif',
    )

    # XYZT
    hyman_hela = ('12qOGxfBrnzrufgbizyTkhHipgRwjSIz-', 'Hyman_HeLa.tif')

    # XYZCT 1344 × 1024 × 1 × 1 × 93
    ome_mitocheck = ('1B9d8Yw_lidZg43U3VZAoalVHf9eHbCS7', '00001_01.ome.tiff')

    # XYZCT 160 × 220 × 8 × 2 × 12
    ome_spim = ('1BG6jCZGLEs1LDxKXjMqF0aV-iiqlushk', 'SPIM-ModuloAlongZ.ome.tiff')


def download_from_gdrive(
    id, name, dest_folder=datasets_folder, overwrite=False, unzip=False
):
    """Download a file from Google Drive by its file ID.

    Parameters
    ----------
    id : str
        Google Drive file ID.
    name : str
        Filename to save the downloaded file as.
    dest_folder : str
        Destination folder for the downloaded file.
    overwrite : bool
        If True, re-downloads even if the file already exists.
    unzip : bool
        If True, unzips the downloaded file into ``dest_folder``.

    Returns
    -------
    output_path : str or None
        Path to the downloaded file, or None if the file already existed.
    """
    try:
        os.makedirs(dest_folder)
    except Exception:
        pass

    url = f'https://drive.google.com/uc?id={id}'
    output_path = join(dest_folder, name)
    if overwrite or not exists(output_path):
        lprint(f"Downloading file {output_path} as it does not exist yet.")
        gdown.download(url, output_path, quiet=False)

        if unzip:
            lprint(f"Unzipping file {output_path}...")
            zip_ref = zipfile.ZipFile(output_path, 'r')
            # Validate paths to prevent Zip Slip vulnerability
            dest_folder_real = os.path.realpath(dest_folder)
            for member in zip_ref.namelist():
                member_path = os.path.realpath(os.path.join(dest_folder, member))
                if not member_path.startswith(dest_folder_real + os.sep):
                    raise ValueError(f"Attempted path traversal in zip file: {member}")
            zip_ref.extractall(dest_folder)
            zip_ref.close()
            # os.remove(output_path)

        return output_path
    else:
        lprint(f"Not downloading file {output_path} as it already exists.")
        return None


def download_all_examples():
    """Download all example images to the local cache."""
    for example in examples_single:
        print(download_from_gdrive(*example.value))


def downloaded_example(substring):
    """Download a specific example whose filename contains the given substring.

    Parameters
    ----------
    substring : str
        Substring to match against example filenames.
    """
    for example in examples_single:
        if substring in example.value[1]:
            print(download_from_gdrive(*example.value))
