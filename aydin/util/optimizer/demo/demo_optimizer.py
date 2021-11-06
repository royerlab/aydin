# flake8: noqa
from math import sin
import numpy

from aydin.util.optimizer.optimizer import Optimizer
from aydin.util.log.log import Log, lprint


def enable_log():
    # DO NOT REMOVE, FOR DEBUGGING PUPROSES!
    Log.enable_output = True
    Log.set_log_max_depth(5)
    pass


def demo_optimizer_mccormik():
    enable_log()

    # function to optimise:
    def mccormik(x, y):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        return -(sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1) - 1.9133

    x, v = Optimizer().optimize(
        mccormik, bounds=[(-1.5, 4.0), (-3.0, 4.0)], max_num_evaluations=512
    )

    print("\n Mccormik")
    print(f"RESULT : x,v={x, v}")
    print(f"OPTIMUM:   x={-0.54719, -1.54719}")

    assert v > -1e-4

    assert numpy.allclose(x, [-0.54719, -1.54719], atol=1e-2)


def demo_optimizer_himmelblau():
    enable_log()

    # function to optimise:
    def himmelblau(x, y):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        return -((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2)

    x, v = Optimizer().optimize(
        himmelblau, bounds=[(-5.0, 5.0), (-5.0, 5.0)], max_num_evaluations=512
    )

    print("\n Himmelblau")
    print(f"RESULT: x,v={x, v}")
    print(
        f"OPTIMUM:   x={3, 2}, or {-2.805, 3.131}, or {-3.779, -3.283}, or {3.584, -1.848}"
    )

    assert v > -1e-3


def demo_optimizer_beale():
    enable_log()

    # function to optimise:
    def beale(x, y):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        return -(
            (1.5 - x + x * y) ** 2
            + (2.25 - x + x * (y ** 2)) ** 2
            + (2.625 - x + x * y ** 3) ** 2
        )

    x, v = Optimizer().optimize(
        beale, bounds=[(-4.5, 4.5), (-4.5, 4.5)], max_num_evaluations=512
    )

    print("\n Beale")
    print(f"RESULT : x,v={x, v}")
    print(f"OPTIMUM:   x={3, 0.5}")

    assert v > -1e-3

    # assert numpy.allclose(x, [3, 0.5], atol=1e-1)


def demo_optimizer_matyas():
    enable_log()

    # function to optimise:
    def matyas(x, y):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        return -(0.26 * (x ** 2 + y ** 2) - 0.48 * x * y)

    x, v = Optimizer().optimize(
        matyas, bounds=[(-10.0, 10.0), (-10.0, 10.0)], max_num_evaluations=512
    )

    print("\n Matyas")
    print(f"RESULT : x,v={x, v}")
    print(f"OPTIMUM:   x={0, 0}")

    assert v > -1e-3

    assert numpy.allclose(x, [0, 0], atol=1e-2)


def demo_optimizer_goldsteinprice():
    enable_log()

    # function to optimise:
    def goldsteinprice(x, y):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        return (
            1
            + (x + y + 1) ** 2
            * (19 - 14 * x + 3 * (x ** 2) - 14 * y + 6 * x * y + 3 * (y ** 2))
        ) * (
            (30 + (2 * x - 3 * y) ** 2)
            * (18 - 32 * x + 12 * (x ** 2) + 48 * y - 36 * x * y + 27 * (y ** 2))
        )

    x, v = Optimizer().optimize(
        goldsteinprice, bounds=[(-2.0, 2.0), (-2.0, 2.0)], max_num_evaluations=512
    )

    print("\n Goldstein & Price")
    print(f"RESULT : x,v={x, v}")
    print(f"OPTIMUM:   x={0, -1}")

    assert v > -1e-3

    # assert numpy.allclose(x, [0, -1], atol=1e-1)


def demo_optimizer_rosenbrock():
    enable_log()

    # function to optimise:
    def rosenbrock(x, y):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        return -(100 * (y - x ** 2) ** 2 + (1 - x) ** 2)

    x, v = Optimizer().optimize(
        rosenbrock, bounds=[(-10.0, 10.0), (-10.0, 10.0)], max_num_evaluations=512
    )

    print("\n Rosenbrock")
    print(f"RESULT : x,v={x, v}")
    print(f"OPTIMUM: x  ={1, 1}")

    assert v > -1e-1


#    assert numpy.allclose(x, [1, 1], atol=1e-1)


def demo_optimizer_sphere():
    enable_log()

    # function to optimise:
    def sphere(x, y):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        return -((x - 0.54321) ** 2 + (y + 0.332211) ** 2)

    x, v = Optimizer().optimize(
        sphere, bounds=[(-10.0, 10.0), (-10.0, 10.0)], max_num_evaluations=512
    )

    print("\n Sphere")
    print(f"RESULT : x,v={x, v}")
    print(f"OPTIMUM:   x={0.54321, -0.332211}")

    assert v > -1e-3

    assert numpy.allclose(x, [0.54321, -0.332211], atol=1e-2)


def demo_optimizer_booth():
    enable_log()

    # function to optimise:
    def booth(x, y):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        return -((x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2)

    x, v = Optimizer().optimize(
        booth, bounds=[(-10.0, 10.0), (-10.0, 10.0)], max_num_evaluations=512
    )

    print("\n Booth")
    print(f"RESULT : x,v={x, v}")
    print(f"OPTIMUM:   x={1, 3}")

    assert v > -1e-1

    assert numpy.allclose(x, [1, 3], atol=1e-2)


def demo_optimizer():
    demo_optimizer_beale()
    demo_optimizer_mccormik()
    demo_optimizer_booth()
    demo_optimizer_sphere()
    demo_optimizer_rosenbrock()
    demo_optimizer_goldsteinprice()
    demo_optimizer_matyas()
    demo_optimizer_himmelblau()


if __name__ == "__main__":
    demo_optimizer()
