import os
import zipfile
from os.path import join, exists

import gdown

from pitl.io.folders import get_cache_folder

datadir = join(get_cache_folder(), 'data')

try:
    os.makedirs(datadir)
except:
    pass


class example_datasets():

    @staticmethod
    def get_list():
        return [example for (key,example) in example_datasets.__dict__.items() if not '__' in key and not 'get_' in key]

    @staticmethod
    def get_path(id, name):
        return join(datadir, name)

    # XY natural images:
    generic_crowd = ('13UHK8MjhBviv31mAW2isdG4G-aGaNJIj','crowd.tif')
    generic_mandrill = ('1B33ELiFuCV0OJ6IHh7Ix9lvImwI_QkR-','mandrill.tif')

    # XYC 3C (RGB)
    celldiv =  ('120w8j2XgJgwD0w0nqX-Gd0C4Qi_gJ8oO', 'Example-noisy1.png')

    # XY
    fmdd_hv115 = ('12C3_nW_wCFftKN0_XmGNoe-v72mp31p-','HV115_P0500510039.png')
    fmdd_hv110 = ('1B6WMgiaaUozgqwvHQtM0NwTUpmKuauKO','HV110_P0500510004.png')

    # XYZ
    keller_dmel = ('12DCAlDRSiTyGDSD7p06nk17GO3ztHg-Q', 'SPC0_TM0132_CM0_CM1_CHN00_CHN01.fusedStack.tif')

    janelia_flybrain= ('12Z6W_f3TqCsl_okKmaLcVUBgS6xEvdjj','Flybrain_3ch_mediumSize.tif')

    # XYZT
    hyman_hela = ('12qOGxfBrnzrufgbizyTkhHipgRwjSIz-', 'Hyman_HeLa.tif')
    pourquie_elec = ('12VMZ6nphV9D40xiKYK6GpH9VghZM3IJC','20190203_p4_electroporated.tif')
    pourquie_quail= ('12SEwGxgFuCd9Oz6c8TbwOcXjOCXUx-aB','20181228_p1_quail.tif')
    gardner_org = ('12MiulopOAa18o2haKZPfyA2piK7x9l3N','405Col4Lm_488EpCAM_INS568_647GCGold_ImmSol_63x___-08.czi')

    # XYZCT 1344 × 1024 × 1 × 1 × 93
    ome_mitocheck = ( '1B9d8Yw_lidZg43U3VZAoalVHf9eHbCS7', '00001_01.ome.tiff')

    # XYZCT 160 × 220 × 8 × 2 × 12
    ome_spim = ( '1BG6jCZGLEs1LDxKXjMqF0aV-iiqlushk', 'SPIM-ModuloAlongZ.ome.tiff')


class zipped_datasets():

    @staticmethod
    def get_list():
        return [example for (key,example) in zipped_datasets.__dict__.items() if not '__' in key and not 'get_' in key]

    @staticmethod
    def get_path(id, name):
        return join(datadir, os.path.splitext(name)[0])

    care_tribolium =  ('1BVNU-y9NJdNzkmsZcH8-2nhdhlRd4Mcw', 'tribolium.zip')




def download_from_gdrive(id, name, dest_folder=datadir, overwrite=False, unzip=False):

    try:
        os.makedirs(dest_folder)
    except:
        pass

    url = f'https://drive.google.com/uc?id={id}'
    output_path = join(dest_folder, name)
    if overwrite or not exists(output_path):
        print(f"Downloading file {output_path} as it does not exist yet.")
        gdown.download(url, output_path, quiet=False)

        if unzip:
            print(f"Unzipping file {output_path}...")
            zip_ref = zipfile.ZipFile(output_path, 'r')
            zip_ref.extractall(dest_folder)
            zip_ref.close()
            #os.remove(output_path)

        return output_path
    else:
        print(f"Not downloading file {output_path} as it already exists.")
        return None


def download_all_examples():

    for example in example_datasets.get_list():
        print(download_from_gdrive(*example))

    for example in zipped_datasets.get_list():
        download_from_gdrive(*example, dest_folder=join(datadir,os.path.splitext(example[1])[0]), unzip=True)


def downloaded_example(substring):
    for example in example_datasets.get_list():
        if substring in example[1]:
            print(download_from_gdrive(*example))


def downloaded_zipped_example(substring):
    for example in zipped_datasets.get_list():
        if substring in example[1]:
            download_from_gdrive(*example, dest_folder=join(datadir,os.path.splitext(example[1])[0]), unzip=True)

