from pathlib import Path

import imageio
import numpy
import pims
import skimage
from czifile import czifile
from tifffile import tifffile


def imread(path):

    extension = Path(path).suffix

    is_tiff = '.tif'  in extension or '.tiff' in extension
    is_ome  = is_tiff and '.ome.tif' in path
    is_czi  = '.czi'  in extension
    is_png  = '.png'  in extension
    is_zarr = '.zarr' in extension
    is_nd2  = '.nd2'  in extension
    is_npy  = '.npy'  in extension

    array = None

    try:
        try:
            if is_tiff:
                print(f"Reading file {path} as TIFF file")
                array = tifffile.imread(path)
            elif is_czi:
                print(f"Reading file {path} as CZI file")
                array = czifile.imread(path)
            elif is_png:
                print(f"Reading file {path} as PNG file")
                array = skimage.io.imread(path)
            else:
                print(f"Reading file {path} using skimage imread")
                array = skimage.imread(path)

        except:
            print(f"Reading file {path} using pims")
            array = pims.open(path)
    except:
        print(f"Could not read file {path} !")
        array = None

    #pims


    metadata={}

    # Remove single-dimensional entries from the array shape.
    array = numpy.squeeze(array)

    return (array, metadata)




