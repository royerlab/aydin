import numpy as np
from napari.util import app_context

from skimage.data import camera
from skimage.exposure import rescale_intensity

from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures


def demo_multiscale_convolutions_2d():
    image = camera().astype(np.float32)  # [0:3,0:3]
    image = np.zeros((7,9))
    image[3, 4] = 1
    #image[1, 1] = 1
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    msf = MultiscaleConvolutionalFeatures(exclude_center=True,
                                          kernel_widths=[3, 3],
                                          kernel_scales=[1, 3],
                                          )

    features = np.moveaxis(msf.compute(image), -1, 0)

    print(image)
    print(features)
    print(features.shape)

    from napari import ViewerApp
    with app_context():
        viewer = ViewerApp()
        layer = viewer.add_image(rescale_intensity(features, in_range='image', out_range=(0, 1)), name='image')
        #layer.colormap('divergent')


def demo_multiscale_convolutions_3d():
    # image = np.random.rand(48, 773, 665)*0;
    image = np.ones((48, 773, 665), dtype=np.float32)

    msf = MultiscaleConvolutionalFeatures(exclude_center=True)

    features = msf.compute(image)

    print(image)
    print(features)
    print(features.shape)


demo_multiscale_convolutions_2d()
