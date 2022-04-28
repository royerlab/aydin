from aydin.io.datasets import characters
from aydin.util.edge_filter.fast_edge_filter import fast_edge_filter
from aydin.util.log.log import lsection, Log


def demo_fast_edge(display=True):

    Log.enable_output = True
    Log.set_log_max_depth(5)

    image = characters()

    with lsection(f"Computing edge filter for image of shape: {image.shape}"):
        image_sobel = fast_edge_filter(image)

    with lsection(f"Computing edge filter for image of shape: {image.shape}, again"):
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
