"""Utility for downloading and extracting zip resources from URLs."""

import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from aydin.util.log.log import lprint


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
    lprint("relativepath= ", relative_path_to_zip)

    # Check if target resource already exists, retrieve the resource if not exists
    if os.path.exists(relative_path_to_zip[:-4]):
        lprint("Resource already exists, nothing to download")
    else:
        urlretrieve(url, relative_path_to_zip)
        # Extract the content
        with zipfile.ZipFile(relative_path_to_zip, "r") as zip_ref:
            # Validate paths to prevent Zip Slip vulnerability
            targetdir_real = os.path.realpath(str(targetdir))
            for member in zip_ref.namelist():
                member_path = os.path.realpath(os.path.join(str(targetdir), member))
                if not member_path.startswith(targetdir_real + os.sep):
                    raise ValueError(f"Attempted path traversal in zip file: {member}")
            zip_ref.extractall(str(targetdir))

        # Delete zip file
        Path.unlink(Path(relative_path_to_zip))
