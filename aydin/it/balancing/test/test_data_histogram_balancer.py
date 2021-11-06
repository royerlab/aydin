import numpy

from aydin.io.datasets import camera, normalise
from aydin.it.balancing.data_histogram_balancer import DataHistogramBalancer


def test_no_balancing():
    balancer = DataHistogramBalancer(keep_ratio=0.5, balance=False)

    image = normalise(camera().astype(numpy.float32, copy=False)).ravel()

    balancer.calibrate(image, batch_length=16)

    entries = [image[i : i + 16] for i in range(0, image.size, 16)]

    balancer.initialise(len(entries))

    count_accepted = 0

    for entry in entries:

        accepted = balancer.add_entry(entry)

        if accepted:
            count_accepted += 1

    print(f"accepted: {count_accepted} / {len(entries)}")
    assert (0.5 - count_accepted / len(entries)) < 0.01


def test_balancing():
    balancer = DataHistogramBalancer(keep_ratio=0.5, balance=True)

    image = normalise(camera().astype(numpy.float32)).ravel()

    balancer.calibrate(image, batch_length=16)

    entries = [image[i : i + 16] for i in range(0, image.size, 16)]

    balancer.initialise(len(entries))

    count_accepted = 0

    for entry in entries:

        accepted = balancer.add_entry(entry)

        if accepted:
            count_accepted += 1

    print(f"accepted: {count_accepted} / {len(entries)}")
    assert count_accepted / len(entries) < 0.4


def test_multiple_runs():
    balancer = DataHistogramBalancer(keep_ratio=0.5, balance=True)

    image = normalise(camera().astype(numpy.float32)).ravel()

    balancer.calibrate(image, batch_length=16)

    entries = [image[i : i + 16] for i in range(0, image.size, 16)]

    for j in range(10):
        balancer.initialise(len(entries))

        count_accepted = 0

        for entry in entries:

            accepted = balancer.add_entry(entry)

            if accepted:
                count_accepted += 1

        print(f"accepted: {count_accepted} / {len(entries)}")
        assert count_accepted / len(entries) < 0.4
