from aydin.io import io
from aydin.io.datasets import examples_single


def test_metadata_rgb_image():

    rgb_image_path = examples_single.rgbtest.get_path()
    _, metadata = io.imread(rgb_image_path)
    print(metadata)

    assert len(metadata.shape) == 3
    assert metadata.shape[-1] == 3
    assert metadata.axes == 'YXC'
    assert metadata.batch_axes == (False, False, False)
    assert metadata.channel_axes == (False, False, True)
    assert metadata.format == 'png'


def test_metadata_complex_4D_stack():

    image_path = examples_single.royerlab_hcr.get_path()
    array, metadata = io.imread(image_path)

    print(metadata)

    assert metadata.is_folder is False
    assert metadata.extension == 'tif'
    assert metadata.axes == 'ZCYX'
    assert metadata.shape == (75, 3, 444, 331)
    assert metadata.batch_axes == (False, False, False, False)
    assert metadata.channel_axes == (False, True, False, False)
