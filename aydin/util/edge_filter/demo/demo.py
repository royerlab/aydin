"""Demo of the fast edge filter utility.

Demonstrates the ``fast_edge_filter`` function on a characters test
image, applying the Sobel-based edge detection and displaying the
result.
"""

from aydin.io.datasets import characters
from aydin.util.edge_filter.fast_edge_filter import fast_edge_filter
from aydin.util.log.log import Log, asection


def demo_fast_edge(display=True):
    """Compute and optionally display edge-filtered image.

    Parameters
    ----------
    display : bool, optional
        Whether to display results in napari, by default True.
    """

    Log.enable_output = True
    Log.set_log_max_depth(5)

    image = characters()

    with asection(f"Computing edge filter for image of shape: {image.shape}"):
        image_sobel = fast_edge_filter(image)

    with asection(f"Computing edge filter for image of shape: {image.shape}, again"):
        image_sobel = fast_edge_filter(image)

    assert image_sobel is not None

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(image_sobel, name='image_sobel')

        napari.run()


if __name__ == "__main__":
    demo_fast_edge()
