import os
import zipfile
from enum import Enum
from os.path import join, exists
from typing import Optional

import gdown
import numpy
import skimage
from scipy.ndimage import binary_dilation, zoom
from scipy.signal import convolve
from scipy.signal import convolve2d
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
    return rescale_intensity(
        image.astype(numpy.float32, copy=False), in_range='image', out_range=(0, 1)
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
    if seed is not None:
        seed = abs(seed)
        numpy.random.seed(seed)
    noisy = image
    if intensity is not None:
        noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode="gaussian", var=variance, seed=seed, clip=clip)
    noisy = random_noise(noisy, mode="s&p", amount=sap, seed=seed, clip=clip)
    noisy = noisy.astype(dtype, copy=False)
    return noisy


def add_blur_2d(image, dz=0):
    psf = SimpleMicroscopePSF()
    psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)
    psf_kernel = psf_xyz_array[dz]
    psf_kernel /= psf_kernel.sum()
    return convolve2d(image, psf_kernel, mode='same', boundary="symm"), psf_kernel


def add_blur_3d(image, xy_size=17, z_size=17):
    psf = SimpleMicroscopePSF()
    psf_xyz_array = psf.generate_xyz_psf(
        dxy=0.406, dz=0.406, xy_size=xy_size, z_size=z_size
    )
    psf_kernel = psf_xyz_array
    psf_kernel /= psf_kernel.sum()
    return convolve(image, psf_kernel, mode='same'), psf_kernel


# Convenience shortcuts:


def lizard():
    return examples_single.generic_lizard.get_array()


def camera():
    return skimage.data.camera().astype(numpy.float32, copy=False)


def newyork():
    return examples_single.generic_newyork.get_array()


def small_newyork():
    return zoom(newyork(), zoom=0.5)


def cropped_newyork(crop_amount=256):
    crop_amount = min(crop_amount, 500)
    return newyork()[crop_amount:-crop_amount, crop_amount:-crop_amount]


def pollen():
    return examples_single.generic_pollen.get_array()


def scafoldings():
    return examples_single.generic_scafoldings.get_array()


def characters():
    return 1 - examples_single.generic_characters.get_array()


def andromeda():
    return examples_single.generic_andromeda.get_array()


def dots():
    image = numpy.random.rand(512, 512) < 0.005  # andromeda()#[256:-256, 256:-256]
    image = 0.8 * binary_dilation(image).astype(numpy.float32, copy=False)
    image[0:256, 0:256] += 0.1
    image.clip(0, 1)
    return image


def rgbtest():
    return examples_single.rgbtest.get_array()


def dmel():
    return examples_single.keller_dmel.get_array()[23]


class examples_single(Enum):
    def get_path(self):
        download_from_gdrive(*self.value, datasets_folder)
        return join(datasets_folder, self.value[1])

    def get_array(self):
        array, _ = io.imread(self.get_path())
        return array

    fountain = ('1JP-_j-6U7J1gNc9IZCZ_GsgXTcybmZgS', 'fountain.png')
    monalisa = ('15T3oTCyz7ugnPLTsKc0a9NT17g9GJsO_', 'monalisa.png')
    gauss_noisy = ('17e_ECJA7DUQGu9JELbTkAKbOVVE9olHN', 'Gauss_noisy.png')
    periodic_noise = ('1HfwF6gnzHFFdJ-tozllU_h14vNk9GZOG', 'periodic_noise.png')
    brown_chessboard_gray = (
        '1gnqwhZ7HrRaScj6QF_P2Pl_6WAcLzCgR',
        'Brown_SIDD_chessboard_gray.png',
    )

    # XY natural images (2D monochrome):
    generic_crowd = ('13UHK8MjhBviv31mAW2isdG4G-aGaNJIj', 'crowd.tif')
    generic_mandrill = ('1B33ELiFuCV0OJ6IHh7Ix9lvImwI_QkR-', 'mandrill.tif')
    generic_newyork = ('15Nuu_NU3iNuoPRmpFbrGIY0VT0iCmuKu', 'newyork.png')
    generic_lizard = ('1GUc6jy5QH5DaiUskCrPrf64YBOLzT6j1', 'lizard.png')
    generic_pollen = ('1S0o2NWtD1shB5DfGRIqOFxTLOi8cHQD-', 'pollen.png')
    generic_scafoldings = ('1ZiWhHnkuaQH-BS8B71y00wkN1Ylo38nY', 'scafoldings.png')
    generic_andromeda = ('1Zl3DtkwUlZSbvpxGILexiIoLW1JOdJh8', 'andromeda.png')

    # Characters (2D monochrome, inverted):
    generic_characters = ('1ZWkHFI2iddKa9qv6tft4QZlCoDS5fLMK', 'characters.jpg')

    # XYC (RGB)
    rgbtest = ('1KvhcGBqEQ5N9mwxHwy14NVp8OJ-9GCsH', 'rgbtest.png')

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
            zip_ref.extractall(dest_folder)
            zip_ref.close()
            # os.remove(output_path)

        return output_path
    else:
        lprint(f"Not downloading file {output_path} as it already exists.")
        return None


def download_all_examples():
    for example in examples_single:
        print(download_from_gdrive(*example.value))


def downloaded_example(substring):
    for example in examples_single.get_list():
        if substring in example.value[1]:
            print(download_from_gdrive(*example.value))
