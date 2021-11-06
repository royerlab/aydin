import itertools
import math
import traceback
from copy import copy
from typing import Callable, List, Tuple, Union
import numpy
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError, norm
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize

from aydin.util.log.log import lprint, lsection


class Optimizer:
    def __init__(self):
        pass

    def optimize(
        self,
        function: Callable,
        bounds: List[Union[Tuple[int, ...], Tuple[float, ...]]],
        init_strategies: str = 'corners+centers+random',
        exploration_rate: float = 0.4,
        patience: int = 64,
        max_num_evaluations: int = 128,
        num_interpolated_evaluations: int = 128,
        workers: int = -1,
    ):
        """
        Optimizes (maximizes) a given function by alternating between optimisation
        of a proxy function obtrained through interpolation, and exploration of the
        least sampled regions of the optimisation domain.

        Parameters
        ----------
        function: Callable
            Function to optimize.

        bounds: List[Union[Tuple[int, ...], Tuple[float, ...]]]
            Bounds for function parameters

        init_strategies: str
            Initialisation strategies. Can contain: 'corners', 'centers', and 'random'

        exploration_rate: float
            Rate at which to explore

        max_num_evaluations: int
            Maximum number of evaluations of the /a priori/ costly given function.

        num_interpolated_evaluations: int
            Max number of evaluations of the inyterpolated function.

        workers: int
            Number of workers, if -1 the maximum is used.


        Returns
        -------
        optimal_point, optimal_value
        optimal_point: Optimal value for parameters
        optimal_value: Corresponding function value

        """
        # First we figure out the dimensionality of the problem:
        n = len(bounds)

        # Save Function:
        self.function = function

        # Save bounds:
        self.bounds = bounds

        # Second, we allocate the array that stores the evaluations:
        self.x = []
        self.y = []

        # We keep track here of the best evaluation:
        self.best_point = None
        self.best_value = -math.inf

        # First we initialise with some points on the corners:
        if 'corners' in init_strategies:
            with lsection("Evaluating function at corners"):

                if 'centers' in init_strategies:
                    init_grid = tuple((u, 0.5 * (u + v), v) for u, v in bounds)
                else:
                    init_grid = copy(bounds)

                point_list = list(itertools.product(*init_grid))
                self._add_points(point_list, workers=workers, display_points=True)

        # First we initialise with some random points:
        if 'random' in init_strategies:
            with lsection("Evaluating function at random points"):
                point_list = list(self._random_sample() for _ in range(min(4, 2 * n)))
                self._add_points(point_list, workers=workers)

        # Foir how long did we not see an improvement?
        self.since_last_best = 0

        # This is the main loop that evaluates the function:
        with lsection(
            f"Optimizing function with at most {max_num_evaluations} function evaluations within: {bounds}"
        ):
            for i in range(max_num_evaluations):
                # lprint(f"Evaluation #{i}")
                # lprint(f"x={x}")

                # Given the existing points, we can build the interpolating function:
                try:
                    self.interpolator = RBFInterpolator(
                        y=numpy.stack(self.x),
                        d=numpy.stack(self.y),
                        neighbors=8 if len(self.x) < 8 else 4 * n,
                        smoothing=abs(numpy.random.normal(0, 1)) ** 0.5,
                    )

                    # From time to time we just pick points far from all other points:
                    do_explore = numpy.random.random() < exploration_rate

                    # using the interpolator we can quickly search for the best value:
                    new_point = self._delegated_optimizer(
                        do_explore=do_explore,
                        num_evaluations=num_interpolated_evaluations,
                    )

                    # We add that point to the list of points:
                    has_new_best = self._add_points([new_point])

                    lprint(
                        f"{i}{'!' if has_new_best else' '}: {' Exploring' if do_explore else 'Optimizing'}, Best point: {self.best_point}, best value: {self.best_value}, new point: {new_point})"
                    )

                    # Are we running out of patience?
                    if self.since_last_best > patience:
                        # If yes we stop searching:
                        lprint(
                            f"Run out of patience: {self.since_last_best} > {patience} !"
                        )
                        break

                except LinAlgError:
                    lprint("Error while optimizing, let's stop training now!")
                    lprint(f"x={self.x}")
                    traceback.print_exc()
                    break

        lprint(f"Best point: {self.best_point}, best value: {self.best_value}")

        return self.best_point, self.best_value

    def _add_points(self, point_list: List, workers=-1, display_points=False):

        # Normalise points:
        point_list = list(
            numpy.array(point, dtype=numpy.float32) for point in point_list
        )

        def _function(*_point):
            _value = self.function(*_point)
            if display_points:
                lprint(f"New point: {_point} -> {_value}")
            return _value

        # Evaluate function in parallel:
        values = Parallel(n_jobs=workers, backend='threading')(
            delayed(_function)(*point) for point in point_list
        )

        # to return:
        has_new_best = False

        # Going through the list of points:
        for new_value, new_point in zip(values, point_list):

            # Replace NaNs or other weird floats with something better:
            new_value = numpy.nan_to_num(new_value, neginf=-1e6, posinf=-1e6, nan=-1e6)

            # And add this new point to the list:
            self.x.append(new_point)
            self.y.append(new_value)
            # We keep track of the last best evaluation:
            if new_value > self.best_value:
                has_new_best = True
                self.since_last_best = 0
                self.best_value = new_value
                self.best_point = new_point
            else:
                self.since_last_best += 1

        return has_new_best

    def _delegated_optimizer(
        self, do_explore: bool, num_evaluations: int = 128, workers: int = -1
    ):
        # If we ran out of evaluations (recursive call!), then let's return immediately with None:
        if num_evaluations <= 0:
            return None

        # First we figure out the dimensionality of the problem:
        n = len(self.bounds)

        # This is the function to optimize:
        def function(point):

            value = 0

            fallback_exploration = False
            if not do_explore:
                # We compute interpolated value:
                try:
                    # interpolation value:
                    value += self.interpolator(point.reshape(n, -1).T)
                except Exception as e:
                    lprint(f"Exception: {e}")
                    # If there is an issue with interpolation, we fallback on exploration:
                    fallback_exploration = True

            if do_explore or fallback_exploration:
                # point coordinates translated for usage with the kd-tree:
                point_for_tree = numpy.array(point)[numpy.newaxis, ...]

                # We collect neighbors:
                distances, indices = self.interpolator._tree.query(
                    point_for_tree, k=1, workers=workers
                )
                indices = indices.flatten()

                # Corresponding point:
                neighboor = self.x[indices[0]]

                # Vector from neighboor to point:
                vector = neighboor - point

                # We add the lipschitz value to the interpolated value:
                value += norm(vector)

            return value

        if do_explore:
            # Random optimizer is great to avoid getting stuck:
            point = self._random_optimizer(function)
        else:
            # Fast minimisation with Neadler-Mead helps get closer to the optimum,
            # here we don't need randomness as much:
            result = minimize(
                lambda x_: -function(x_),  # this is a minimiser
                x0=self.best_point,
                method='Nelder-Mead',
                bounds=self.bounds,
                options={'maxiter': num_evaluations},
            )
            point = result.x

        # the RBF interpolator hates it when a point occurs multiple times,
        # we clean up:
        while True:
            if _is_in(point, self.x) or point is None:
                # If we get a point we already have, lets pick a point at random:
                # lprint(
                #     f"Point {point} already suggested (or None), trying something else..."
                # )
                try:
                    if point is None:
                        # Best point is None, let's pick a random point:
                        point = self._random_sample()
                    else:
                        # Add noise to current point
                        point = self._add_noise(point, sigma=0.01)
                    # lprint(f"point: {point}")
                except RecursionError:
                    # Fail safe in case we recurse too much:
                    point = self._random_sample()
            else:
                # we are good to go...
                break

        return point

    def _compute_lipschitz(self):
        # point coordinates translated for usage with teh kd-tree:
        point_for_tree = numpy.array(self.best_point)[numpy.newaxis, ...]

        # We collect neighbors:
        distances, indices = self.interpolator._tree.query(
            point_for_tree, k=2 * len(self.best_point), workers=-1
        )
        indices = indices.flatten()
        length = len(indices)

        # enumerate all slops:
        slopes_list = []
        for u in range(0, length - 1):
            for v in range(u + 1, length):
                point_u = self.x[indices[u]]
                point_v = self.x[indices[v]]

                value_u = self.y[indices[u]]
                value_v = self.y[indices[v]]

                slopes = numpy.abs((value_v - value_u) / (point_v - point_u))

                slopes_list.append(slopes)

        # convert to array:
        slopes_array = numpy.stack(slopes_list)

        # We get rid of weird floats:
        slopes_array = numpy.nan_to_num(slopes_array, neginf=0.0, posinf=0.0, nan=0)

        # We compute the median:
        lipschitz = numpy.median(slopes_array, axis=0)

        # make sure that there are no zeroes:
        lipschitz[lipschitz == 0] = numpy.min(lipschitz)

        return lipschitz

    def _random_optimizer(self, function, num_evaluations: int = 128):
        # We keep track of the best values:
        best_point = None
        best_value = -math.inf

        # This is a random optimizer
        for i in range(num_evaluations):
            # We pick a random point:
            point = self._random_sample()

            # evaluate function at point:
            value = function(point)

            # Replace NaNs or other weird floats with something better:
            value = numpy.nan_to_num(value, neginf=-1e6, posinf=-1e6, nan=-1e6)

            # We check if the value is better:
            if value > best_value:
                best_value = value
                best_point = point

        return best_point

    def _random_sample(self):
        # First we figure out the dimensionality of the problem:
        n = len(self.bounds)

        # List to store coordinates
        point = []

        # Loop through each coordinate:
        for i in range(n):
            min_r, max_r = self.bounds[i]

            # int coodinates:
            if type(min_r) is int or type(min_r) in [numpy.int32, numpy.int64]:
                coord = numpy.random.randint(min_r, max_r)

            # float coordinates:
            elif type(min_r) is float or type(min_r) in [numpy.float32, numpy.float64]:
                coord = numpy.random.uniform(min_r, max_r)

            point.append(coord)

        return numpy.array(point)

    def _add_noise(self, point, sigma: float = 1e-6):
        # range widths:
        widths = numpy.array(tuple(abs(float(u - v)) for u, v in self.bounds))

        # mins and maxes :
        mins = numpy.array(tuple(float(min(u, v)) for u, v in self.bounds))
        maxs = numpy.array(tuple(float(max(u, v)) for u, v in self.bounds))

        # List to store coordinates
        point += numpy.random.normal(0, sigma, point.shape) * 0.5 * widths

        # clip to range:
        point = numpy.maximum(point, mins)
        point = numpy.minimum(point, maxs)

        return numpy.array(point)


def _is_in(array, array_list):
    for a in array_list:
        # RBF interpolator hates it when two coordinates of two points have the same values...
        if (a == array).any():
            return True

    return False
