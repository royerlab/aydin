"""Example datasets for testing and demonstration.

This module provides convenience functions to download, cache, and load
example images from Zenodo and original source URLs for testing Aydin
denoising algorithms. It also includes utility functions for adding
synthetic noise and blur.

Datasets are hosted on Zenodo (DOI: 10.5281/zenodo.18686988).
"""

import hashlib
import json
import os
from enum import Enum
from os.path import exists, join
from pathlib import Path
from typing import Optional

import numpy
import requests
from scipy.ndimage import binary_dilation, zoom
from scipy.signal import convolve, convolve2d
from skimage.exposure import rescale_intensity
from skimage.util import random_noise

from aydin.io import io
from aydin.io.folders import get_cache_folder
from aydin.util.log.log import aprint
from aydin.util.psf.simple_microscope_psf import SimpleMicroscopePSF

datasets_folder = join(get_cache_folder(), 'data')

os.makedirs(datasets_folder, exist_ok=True)

# Zenodo record ID for the Aydin example datasets (v1.1.0)
ZENODO_RECORD_ID = "18686988"

# Load checksums from the JSON file shipped with the package
_checksums_path = Path(__file__).parent / 'dataset_checksums.json'
try:
    with open(_checksums_path) as _f:
        CHECKSUMS = json.load(_f)
except FileNotFoundError:
    import warnings

    warnings.warn(
        f"Checksum file not found at {_checksums_path}. "
        "Downloads will proceed without integrity verification.",
        stacklevel=1,
    )
    CHECKSUMS = {}


def _verify_file(path, filename):
    """Verify a downloaded file against known checksums.

    Parameters
    ----------
    path : str
        Path to the downloaded file.
    filename : str
        Filename key in the CHECKSUMS dict.

    Raises
    ------
    IOError
        If the file size or SHA-256 hash does not match.
    """
    if filename not in CHECKSUMS:
        return  # No checksum available, skip verification

    expected = CHECKSUMS[filename]
    actual_size = os.path.getsize(path)
    if actual_size != expected["size"]:
        os.remove(path)
        raise IOError(
            f"Size mismatch for {filename}: "
            f"expected {expected['size']}, got {actual_size}"
        )

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    actual_hash = h.hexdigest()
    if actual_hash != expected["sha256"]:
        os.remove(path)
        raise IOError(
            f"SHA-256 mismatch for {filename}: "
            f"expected {expected['sha256'][:16]}..., "
            f"got {actual_hash[:16]}..."
        )


def _download_file(url, filename, dest_folder=datasets_folder, overwrite=False):
    """Download a file from a URL with streaming and checksum verification.

    Parameters
    ----------
    url : str
        URL to download from.
    filename : str
        Filename to save as.
    dest_folder : str
        Destination folder.
    overwrite : bool
        If True, re-downloads even if the file already exists.

    Returns
    -------
    output_path : str or None
        Path to the downloaded file, or None if it already existed.
    """
    output_path = join(dest_folder, filename)
    if not overwrite and exists(output_path):
        aprint(f"Not downloading {filename} as it already exists.")
        return None
    os.makedirs(dest_folder, exist_ok=True)
    aprint(f"Downloading {filename} from {url[:80]}...")
    resp = requests.get(url, stream=True, timeout=(30, 300))
    resp.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    _verify_file(output_path, filename)
    return output_path


def download_from_zenodo(filename, dest_folder=datasets_folder, overwrite=False):
    """Download a single file from the Aydin Zenodo dataset record.

    Parameters
    ----------
    filename : str
        Name of the file on Zenodo.
    dest_folder : str
        Destination folder for the downloaded file.
    overwrite : bool
        If True, re-downloads even if the file already exists.

    Returns
    -------
    output_path : str or None
        Path to the downloaded file, or None if it already existed.
    """
    url = (
        f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
        f"/files/{filename}/content"
    )
    return _download_file(url, filename, dest_folder, overwrite)


def download_from_url(url, filename, dest_folder=datasets_folder, overwrite=False):
    """Download a file from a direct URL.

    Parameters
    ----------
    url : str
        Direct download URL.
    filename : str
        Filename to save as.
    dest_folder : str
        Destination folder for the downloaded file.
    overwrite : bool
        If True, re-downloads even if the file already exists.

    Returns
    -------
    output_path : str or None
        Path to the downloaded file, or None if it already existed.
    """
    return _download_file(url, filename, dest_folder, overwrite)


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
    """Load the example lizard image (2D monochrome).

    Returns
    -------
    image : numpy.ndarray
        2D monochrome lizard image.
    """
    return examples_single.generic_lizard.get_array()


def camera():
    """Load the example camera image (2D monochrome).

    Returns
    -------
    image : numpy.ndarray
        2D monochrome camera image.
    """
    return examples_single.generic_camera.get_array()


def newyork():
    """Load the example New York image (2D monochrome).

    Returns
    -------
    image : numpy.ndarray
        2D monochrome New York cityscape image.
    """
    return examples_single.generic_newyork.get_array()


def small_newyork():
    """Load a downscaled (0.5x) version of the New York image.

    Returns
    -------
    image : numpy.ndarray
        2D monochrome New York image downscaled by a factor of 0.5
        using spline interpolation.
    """
    return zoom(newyork(), zoom=0.5)


def cropped_newyork(crop_amount=256):
    """Load a center-cropped version of the New York image.

    Parameters
    ----------
    crop_amount : int, optional
        Number of pixels to crop from each edge, clamped to a
        maximum of 500. Default is 256.

    Returns
    -------
    image : numpy.ndarray
        Center-cropped 2D monochrome New York image.
    """
    crop_amount = min(crop_amount, 500)
    return newyork()[crop_amount:-crop_amount, crop_amount:-crop_amount]


def newyork_noisy():
    """Load the noisy version of the New York image.

    Returns
    -------
    image : numpy.ndarray
        2D monochrome noisy New York image.
    """
    return examples_single.noisy_newyork.get_array()


def pollen():
    """Load the example pollen image (2D monochrome).

    Returns
    -------
    image : numpy.ndarray
        2D monochrome pollen image.
    """
    return examples_single.generic_pollen.get_array()


def scafoldings():
    """Load the example scaffoldings image (2D monochrome).

    Returns
    -------
    image : numpy.ndarray
        2D monochrome scaffoldings image.
    """
    return examples_single.generic_scafoldings.get_array()


def characters():
    """Load the example characters image (2D monochrome, inverted).

    The raw image is inverted so that characters appear bright on a dark
    background.

    Returns
    -------
    image : numpy.ndarray
        2D monochrome inverted characters image.
    """
    return 1 - examples_single.generic_characters.get_array()


def andromeda():
    """Load the example Andromeda galaxy image (2D monochrome).

    Returns
    -------
    image : numpy.ndarray
        2D monochrome Andromeda galaxy image.
    """
    return examples_single.generic_andromeda.get_array()


def dots():
    """Generate a synthetic sparse dots image with non-uniform background.

    Creates a 512x512 image with randomly placed sparse dots (dilated) and
    a brighter quadrant in the upper-left corner.

    Returns
    -------
    image : numpy.ndarray
        2D float32 image of shape (512, 512) with values in [0, 1].
    """
    image = numpy.random.rand(512, 512) < 0.005  # andromeda()#[256:-256, 256:-256]
    image = 0.8 * binary_dilation(image).astype(numpy.float32, copy=False)
    image[0:256, 0:256] += 0.1
    image = image.clip(0, 1)
    return image


def rgbtest():
    """Load the example RGB test image.

    Returns
    -------
    image : numpy.ndarray
        RGB test image with a channel axis.
    """
    return examples_single.rgbtest.get_array()


def dmel():
    """Load a single slice of the Keller Drosophila melanogaster dataset.

    Returns the 24th Z-slice (index 23) from the 3D stack.

    Returns
    -------
    image : numpy.ndarray
        2D slice from the Drosophila melanogaster light-sheet dataset.
    """
    return examples_single.keller_dmel.get_array()[23]


class examples_single(Enum):
    """Enumeration of single example images available for download.

    Each member is a tuple of (source, filename) where source is either
    ``"zenodo"`` for Zenodo-hosted files or a URL for source-URL files.
    Use ``get_path()`` to download and get the local path, or
    ``get_array()`` to directly get the image as a numpy array.
    """

    def get_path(self):
        """Download (if needed) and return the local file path for this example.

        Returns
        -------
        path : str
            Absolute path to the downloaded example image file.
        """
        source, filename = self.value
        if source == "zenodo":
            download_from_zenodo(filename, datasets_folder)
        elif source.startswith("http"):
            download_from_url(source, filename, datasets_folder)
        return join(datasets_folder, filename)

    def get_array(self):
        """Download (if needed), read, and return the image as a numpy array.

        Returns
        -------
        array : numpy.ndarray
            The example image data as a NumPy array.
        """
        array, _ = io.imread(self.get_path())
        return array

    # --- Zenodo-hosted ---

    # XY natural images (2D monochrome):
    generic_camera = ("zenodo", "camera.png")  # CC BY-NC (MIT)
    generic_mandrill = ("zenodo", "mandrill.tif")  # Copyright unknown (USC-SIPI)
    generic_newyork = ("zenodo", "newyork.png")
    generic_lizard = ("zenodo", "lizard.png")  # Non-commercial (Berkeley BSDS300)
    generic_pollen = ("zenodo", "pollen.png")
    generic_scafoldings = ("zenodo", "scafoldings.png")
    generic_andromeda = ("zenodo", "andromeda.png")

    # XY noisy (2D monochrome):
    noisy_fountain = ("zenodo", "fountain.png")
    noisy_newyork = ("zenodo", "newyork_noisy.tif")
    noisy_monalisa = ("zenodo", "monalisa.png")
    noisy_gauss = ("zenodo", "Gauss_noisy.png")
    noisy_brown_chessboard = ("zenodo", "Brown_SIDD_chessboard_gray.png")

    # Patterned noise (2D monochrome)
    periodic_noise = ("zenodo", "periodic_noise.png")

    # Characters (2D monochrome, inverted):
    generic_characters = ("zenodo", "characters.jpg")

    # XYC (RGB)
    rgbtest = ("zenodo", "rgbtest.png")

    # Leonetti datasets:
    leonetti_tm7sf2 = (
        "zenodo",
        "Leonetti_p4B3_1_TM7SF2_PyProcessed_IJClean.tif",
    )
    leonetti_sptssa = (
        "zenodo",
        "Leonetti_p1H10_2_SPTSSA_PyProcessed_IJClean.tif",
    )
    leonetti_snca = (
        "zenodo",
        "Leonetti_p1H8_2_SNCA_PyProcessed_IJClean.tif",
    )
    leonetti_arhgap21 = (
        "zenodo",
        "Leonetti_OC-FOV_ARHGAP21_ENSG00000107863_CID000556_FID00030711_stack.tif",
    )
    leonetti_ankrd11 = (
        "zenodo",
        "Leonetti_OC-FOV_ANKRD11_ENSG00000167522_CID001385_FID00033338_stack.tif",
    )

    # XYZ
    huang_fixed_pattern_noise = ("zenodo", "fixed_pattern_noise.tif")
    keller_dmel = ("zenodo", "SPC0_TM0132_CM0_CM1_CHN00_CHN01.fusedStack.tif")
    janelia_flybrain = ("zenodo", "Flybrain_3ch_mediumSize.tif")
    myers_tribolium = (
        "zenodo",
        "Myers_Tribolium_nGFP_0.1_0.2_0.5_20_13_late.tif",
    )
    royerlab_hcr = (
        "zenodo",
        "Royer_confocal_dragonfly_hcr_drerio_30somite_crop.tif",
    )
    machado_drosophile_egg_chamber = ("zenodo", "C2-DrosophilaEggChamber-small.tif")

    # 2D+t
    cognet_nanotube1 = (
        "zenodo",
        "Cognet_r03-s01-100mW-20ms-175 50xplpeg-173.tif",
    )
    cognet_nanotube_400fps = ("zenodo", "Cognet_1-400fps.tif")
    cognet_nanotube_200fps = ("zenodo", "Cognet_1-200fps.tif")
    cognet_nanotube_100fps = ("zenodo", "Cognet_1-100fps.tif")

    # 3D+t
    maitre_mouse = (
        "zenodo",
        "Maitre_mouse blastocyst_fracking_180124_e3_crop.tif",
    )

    # XYZT
    hyman_hela = ("zenodo", "Hyman_HeLa.tif")

    # --- Source-URL-hosted (externally-hosted OME-TIFF) ---

    # XYZCT 1344 x 1024 x 1 x 1 x 93
    ome_mitocheck = (
        "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/"
        "MitoCheck/00001_01.ome.tiff",
        "00001_01.ome.tiff",
    )

    # XYZCT 160 x 220 x 8 x 2 x 12
    ome_spim = (
        "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/"
        "modulo/SPIM-ModuloAlongZ.ome.tiff",
        "SPIM-ModuloAlongZ.ome.tiff",
    )


def download_all_examples():
    """Download all example images from :class:`examples_single`.

    Files are saved to the ``datasets_folder`` directory inside the
    platform-specific cache folder. Already-downloaded files are skipped.
    """
    for example in examples_single:
        source, filename = example.value
        if source == "zenodo":
            download_from_zenodo(filename, datasets_folder)
        elif source.startswith("http"):
            download_from_url(source, filename, datasets_folder)


def downloaded_example(substring):
    """Download all examples whose filename contains the given substring.

    Iterates over all members of :class:`examples_single` and downloads
    each one whose filename contains ``substring``. If no filenames match,
    nothing is downloaded.

    Parameters
    ----------
    substring : str
        Substring to match against example filenames (case-sensitive).
    """
    for example in examples_single:
        source, filename = example.value
        if substring in filename:
            if source == "zenodo":
                download_from_zenodo(filename, datasets_folder)
            elif source.startswith("http"):
                download_from_url(source, filename, datasets_folder)
