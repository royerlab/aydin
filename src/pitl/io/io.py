from pathlib import Path



def imread(path):

    extension = Path(path).suffix

    dataset = {}

    is_tiff = '.tif' in extension or '.tiff' in extension
    is_ome  = is_tiff and '.ome.tif' in path


    if is_tiff:
        tifimread()

    return dataset




