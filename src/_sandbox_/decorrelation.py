import numpy as np
from scipy import signal

x = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]]
x = np.array(x)
y = [[4, 5], [3, 4]]
y = np.array(y)

print("conv:", signal.convolve2d(x, y, 'full'))

s1 = np.array(x.shape)
s2 = np.array(y.shape)

size = s1 + s2 - 1

fsize = 2 ** np.ceil(np.log2(size)).astype(int)
fslice = tuple([slice(0, int(sz)) for sz in size])

new_x = np.fft.fft2(x, fsize)

new_y = np.fft.fft2(y, fsize)
result = np.fft.ifft2(new_x * new_y)[fslice].copy()

print("fft for my method:", np.array(result.real, np.int32))

print("fft:", np.array(signal.fftconvolve(x, y), np.int32))
