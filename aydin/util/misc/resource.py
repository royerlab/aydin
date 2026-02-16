"""Utility for downloading and extracting zip resources from URLs."""

import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from aydin.util.log.log import aprint


def download_and_extract_zipresource(url, targetdir='.'):
    """Download a zip file from a URL and extract it to a target directory.

    If the extracted resource already exists, the download is skipped.
    The zip file is deleted after successful extraction.

    Parameters
    ----------
    url : str
        URL of the zip resource to download.
    targetdir : str or Path
        Directory to extract the zip contents into. Created if it does
        not exist. Defaults to the current directory.
    """

    # Check if target directory exists, if not create
    targetdir = Path(targetdir)
    if not targetdir.is_dir():
        targetdir.mkdir(parents=True, exist_ok=True)

    # Compute relative path to resource
    relative_path_to_zip = str(targetdir) + '/' + os.path.basename(url)
    aprint("relativepath= ", relative_path_to_zip)

    # Check if target resource already exists, retrieve the resource if not exists
    if os.path.exists(relative_path_to_zip[:-4]):
        aprint("Resource already exists, nothing to download")
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
