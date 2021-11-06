import sys
from urllib.request import urlretrieve
from pathlib import Path
import os
import zipfile


def resource_path(relative_path):
    """
    Method to get resource path from bundle

    Parameters
    ----------
    relative_path : str

    Returns
    -------
    resource path : str

    """
    try:
        base_path = sys._MEIPASS

        return os.path.join(base_path, os.path.basename(relative_path))
    except Exception:
        base_path = os.path.abspath("..")

    return os.path.join(base_path, relative_path)


def download_and_extract_zipresource(url, targetdir='.'):
    """
    Method to download and extract a zipresource from a url.

    Parameters
    ----------
    url : str
    targetdir : str

    """

    # Check if target directory exists, if not create
    targetdir = Path(targetdir)
    if not targetdir.is_dir():
        targetdir.mkdir(parents=True, exist_ok=True)

    # Compute relative path to resource
    relative_path_to_zip = str(targetdir) + '/' + os.path.basename(url)
    print("relativepath= ", relative_path_to_zip)

    # Check if target resource already exists, retrieve the resource if not exists
    if os.path.exists(relative_path_to_zip[:-4]):
        print("Resource already exists, nothing to download")
    else:
        urlretrieve(url, relative_path_to_zip)
        # Extract the content
        with zipfile.ZipFile(relative_path_to_zip, "r") as zip_ref:
            zip_ref.extractall(str(targetdir))

        # Delete zip file
        Path.unlink(Path(relative_path_to_zip))
