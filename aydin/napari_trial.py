import numpy as np
import napari

shape = (82, 65, 967, 516)
data = np.ones(shape)
for i in range(len(data)):
    start = 50+5*i
    strip = slice(start, start + 7)
    data[i,:,:,strip] = 3

with napari.gui_qt():
    viewer = napari.view_image(data, rgb=False, scale=[1, 1, 0.16, 0.16], contrast_limits=(0, 3))
