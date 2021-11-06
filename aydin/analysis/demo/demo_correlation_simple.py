import random
from math import pi, cos, sin

import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.draw import circle, circle_perimeter, line_aa

from aydin.analysis.correlation import correlation, correlation_distance


def discs(shape=(512, 512), radius=10, num=512, fill=True, image=None):
    if image is None:
        image = np.zeros(shape, dtype=np.float)

    for i in range(num):
        x = int(random.uniform(0, shape[0]))
        y = int(random.uniform(0, shape[1]))
        if fill:
            rr, cc = circle(x, y, radius, shape=image.shape)
            image[rr, cc] = 1
        else:
            rr, cc = circle_perimeter(
                x, y, radius, method='bresenham', shape=image.shape
            )
            image[rr, cc] = 1

    return image


def lines(shape=(512, 512), length=10, num=512, fill=False, image=None):
    if image is None:
        image = np.zeros(shape, dtype=np.float)

    for i in range(num):
        xb = int(random.uniform(length, shape[0] - 1 - length))
        yb = int(random.uniform(length, shape[1] - 1 - length))
        angle = random.uniform(0, 2 * pi)
        xe = int(xb + length * cos(angle))
        ye = int(yb + length * sin(angle))
        rr, cc, val = line_aa(xb, yb, xe, ye)
        image[rr, cc] = val

    return image


plot_length = 256


def noise_example():
    image = np.random.random((1500, 1500))
    plt.imshow(image)
    plt.show()

    plt.plot(correlation(image)[0][:32], label='y')
    plt.plot(correlation(image)[1][:32], label='x')
    plt.legend()
    plt.show()

    print("noise  %d,%d " % correlation_distance(image))


def camera_example():
    camera = data.camera()
    plt.imshow(camera)
    plt.show()

    plt.plot(correlation(camera)[0][:plot_length], label='y')
    plt.plot(correlation(camera)[1][:plot_length], label='x')
    plt.legend()
    plt.show()

    print("camera  %d,%d " % correlation_distance(camera))


def astronaut_example():
    astronaut = rgb2gray(data.astronaut())
    plt.imshow(astronaut)
    plt.show()

    plt.plot(correlation(astronaut)[0][:plot_length], label='y')
    plt.plot(correlation(astronaut)[1][:plot_length], label='x')
    plt.legend()
    plt.show()

    print("astronaut  %d,%d " % correlation_distance(astronaut))


def clock_example():
    clock = data.clock()[70:220, 150:270]
    plt.imshow(clock)
    plt.show()

    plt.plot(correlation(clock)[0][:plot_length], label='y')
    plt.plot(correlation(clock)[1][:plot_length], label='x')
    plt.legend()
    plt.show()

    print("clock  %d,%d " % correlation_distance(clock))


def coins_example():
    coins = data.coins()
    plt.imshow(coins)
    plt.show()

    plt.plot(correlation(coins)[0][:plot_length], label='y')
    plt.plot(correlation(coins)[1][:plot_length], label='x')
    plt.legend()
    plt.show()

    print("coins  %d,%d " % correlation_distance(coins))


def discs_example():
    discs1 = discs(shape=(1500, 1500), radius=40, num=90)
    discs2 = discs(shape=(1500, 1500), radius=20, num=160)
    discs3 = discs(shape=(1500, 1500), radius=10, num=420)

    plt.imshow(discs1)
    plt.show()
    plt.imshow(discs2)
    plt.show()
    plt.imshow(discs3)
    plt.show()

    plt.plot(correlation(discs1)[0][:plot_length], label='y 40')
    plt.plot(correlation(discs2)[0][:plot_length], label='y 20')
    plt.plot(correlation(discs3)[0][:plot_length], label='y 10')
    plt.plot(correlation(discs1)[1][:plot_length], label='x 40')
    plt.plot(correlation(discs2)[1][:plot_length], label='x 20')
    plt.plot(correlation(discs3)[1][:plot_length], label='x 10')
    plt.legend()
    plt.show()

    print("discs1  %d,%d " % correlation_distance(discs1))
    print("discs2  %d,%d " % correlation_distance(discs2))
    print("discs3  %d,%d " % correlation_distance(discs3))


def lines_example():
    lines1 = lines(shape=(1500, 1500), length=40, num=3000)
    lines2 = lines(shape=(1500, 1500), length=20, num=6000)
    lines3 = lines(shape=(1500, 1500), length=10, num=12000)

    plt.imshow(lines1)
    plt.show()
    plt.imshow(lines2)
    plt.show()
    plt.imshow(lines3)
    plt.show()

    plt.plot(correlation(lines1)[0][:plot_length], label='40')
    plt.plot(correlation(lines2)[0][:plot_length], label='20')
    plt.plot(correlation(lines3)[0][:plot_length], label='10')
    plt.legend()
    plt.show()

    print("lines1  %d,%d " % correlation_distance(lines1))
    print("lines2  %d,%d " % correlation_distance(lines2))
    print("lines3  %d,%d " % correlation_distance(lines3))


noise_example()
camera_example()
astronaut_example()
clock_example()
coins_example()
discs_example()
lines_example()
