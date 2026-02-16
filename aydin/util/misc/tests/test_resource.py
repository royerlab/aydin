"""Tests for the ZIP resource download and extraction utility."""

import zipfile
from unittest.mock import patch

import numpy as np
import pytest
from tifffile import imread

from aydin.util.misc.resource import download_and_extract_zipresource


@pytest.mark.heavy
def test_download_and_extract_zipresource():
    """Test downloading and extracting a ZIP resource from a remote URL."""
    # Testing execution of the method
    try:
        download_and_extract_zipresource(
            url='http://csbdeep.bioimagecomputing.com/example_data/tribolium.zip',
            targetdir='data',
        )
    except Exception:
        pytest.fail("download_and_extract_zipresource failed", True)

    # Read some files
    x = imread('data/tribolium/train/GT/nGFP_0.1_0.2_0.5_20_13_late.tif').astype(
        np.float32, copy=False
    )
    y = imread('data/tribolium/train/low/nGFP_0.1_0.2_0.5_20_13_late.tif').astype(
        np.float32, copy=False
    )
    z = imread('data/tribolium/test/GT/nGFP_0.1_0.2_0.5_20_14_late.tif').astype(
        np.float32, copy=False
    )
    t = imread('data/tribolium/test/low/nGFP_0.1_0.2_0.5_20_14_late.tif').astype(
        np.float32, copy=False
    )

    # Check if dimensions of images match from different files, we know they should be same
    assert y.shape == x.shape
    assert z.shape == t.shape


def test_download_skip_if_exists(tmp_path):
    """When extracted resource already exists, download should be skipped."""
    # Create the "already extracted" directory
    resource_dir = tmp_path / 'test_resource'
    resource_dir.mkdir()

    with patch('aydin.util.misc.resource.urlretrieve') as mock_retrieve:
        download_and_extract_zipresource(
            url='http://example.com/test_resource.zip',
            targetdir=str(tmp_path),
        )
        # urlretrieve should not have been called
        mock_retrieve.assert_not_called()


def test_download_extracts_and_cleans_zip(tmp_path):
    """Should download, extract, and delete the zip file."""
    import io

    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test_res/file.txt', 'hello')
    zip_bytes = zip_buffer.getvalue()

    def fake_urlretrieve(url, dest):
        """Write pre-built zip bytes to the destination path."""
        with open(dest, 'wb') as f:
            f.write(zip_bytes)

    with patch('aydin.util.misc.resource.urlretrieve', side_effect=fake_urlretrieve):
        download_and_extract_zipresource(
            url='http://example.com/test_res.zip',
            targetdir=str(tmp_path),
        )

    # The extracted directory should exist
    assert (tmp_path / 'test_res' / 'file.txt').exists()
    # The zip file should have been deleted
    assert not (tmp_path / 'test_res.zip').exists()
