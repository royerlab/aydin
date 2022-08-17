import numpy
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.data import astronaut

from aydin.io.datasets import examples_single, add_noise
from aydin.it.transforms.padding import PaddingTransform
from aydin.it.transforms.range import RangeTransform
from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR


def main():
    # gt_image = img_as_float(rgb2gray(astronaut()))
    # noisy_image = img_as_float(rgb2gray(astronaut()))
    #
    # gt_image_f = numpy.abs(fftshift(fft2(gt_image)))
    # noisy_image_f = numpy.abs(fftshift(fft2(gt_image)))
    #
    # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # ax = axes.ravel()
    # ax[0].set_title("Original image")
    # ax[0].imshow(gt_image, cmap='gray')
    # ax[1].set_title("Noisy image")
    # ax[1].imshow(noisy_image, cmap='gray')
    # ax[2].set_title("FFT Original image")
    # ax[2].imshow(gt_image_f, cmap='magma')
    # ax[3].set_title("FFT Noisy image")
    # ax[3].imshow(noisy_image_f, cmap='magma')
    # plt.show()

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.fft import fft2, fftshift
    from skimage import img_as_float
    from skimage.color import rgb2gray
    from skimage.data import astronaut
    from skimage.filters import window

    image = img_as_float(examples_single.generic_lizard.get_array())
    print(image.shape)

    # wimage = img_as_float(examples_single.noisy_newyork.get_array())
    wimage = img_as_float(add_noise(image))

    transforms = [
        {"class": RangeTransform, "kwargs": {}},
        {"class": PaddingTransform, "kwargs": {}},
    ]
    n2s = Noise2SelfFGR(it_transforms=transforms)
    n2s.train(wimage)
    dimage = n2s.denoise(wimage)


    image_f = np.abs(fftshift(fft2(image)))
    wimage_f = np.abs(fftshift(fft2(wimage)))
    dimage_f = np.abs(fftshift(fft2(dimage)))

    # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # ax = axes.ravel()
    # ax[0].set_title("Original image")
    # ax[0].imshow(image, cmap='gray')
    # ax[1].set_title("Windowed image")
    # ax[1].imshow(wimage, cmap='gray')
    # ax[2].set_title("Original FFT (frequency)")
    # ax[2].imshow(np.log(image_f), cmap='magma')
    # ax[3].set_title("Window + FFT (frequency)")
    # ax[3].imshow(np.log(wimage_f), cmap='magma')
    # plt.show()

    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    # Read data from a csv
    # z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
    # z = z_data.values
    #
    # print(z.shape)

    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])



    z = image_f
    sh_0, sh_1 = z.shape
    x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)

    fig.add_trace(go.Surface(z=np.log(image_f), x=x, y=y, colorscale='Viridis', showscale=False),  row=1, col=1)
    fig.add_trace(go.Surface(z=np.log(wimage_f), x=x, y=y, colorscale='Viridis', showscale=False),  row=1, col=2)
    fig.add_trace(go.Surface(z=np.log(dimage_f), x=x, y=y, colorscale='Viridis', showscale=False), row=1, col=3)


    fig.update_layout(title='Fourier Space', autosize=False,
                      width=1800, height=1200,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()


    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    #
    # num_bars = 1024 * 1024
    # x_pos = np.linspace(0, num_bars, num_bars)
    # y_pos = np.linspace(0, num_bars, num_bars)
    # z_pos = [0] * num_bars
    # x_size = np.ones(num_bars)
    # y_size = np.ones(num_bars)
    # z_size = np.linspace(0, 100, num_bars)
    #
    # ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color='aqua')
    # plt.show()


    # setup the figure and axes
    # fig = plt.figure(figsize=(8, 3))
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    #
    # # fake data
    # _x = np.arange(10)
    # _y = np.arange(10)
    # _xx, _yy = np.meshgrid(_x, _y)
    # x, y = _xx.ravel(), _yy.ravel()
    #
    # top = x + y
    # bottom = np.zeros_like(top)
    # width = depth = 1
    #
    # ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    # ax1.set_title('Shaded')
    #
    # ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
    # ax2.set_title('Not Shaded')
    #
    # plt.show()


    # ax2 = fig.add_subplot(122, projection='3d')
    # X, Y = np.meshgrid(np.linspace(0, 1, 1024), np.linspace(0, 1, 1024))
    # ax2.plot_wireframe(X, Y, image, cmap='magma')
    # plt.show()

    # ax2.contour(X, Y, np.log(image_f), 100, zdir='z', offset=0.5, cmap="plasma")




if __name__ == '__main__':
    main()
