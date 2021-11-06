import math
from functools import partial
from typing import Callable, Dict, Union, Tuple, List, Any
import numpy
from skimage.metrics import mean_squared_error

from aydin.util.j_invariance.j_invariant_classic import (
    _j_invariant_loss,
    _product_from_dict,
    _generate_mask,
)
from aydin.util.log.log import lsection, lprint
from aydin.util.optimizer.optimizer import Optimizer


def calibrate_denoiser_smart(
    image,
    denoise_function: Callable,
    denoise_parameters: Dict[str, Union[Tuple[float, float], List[Any]]],
    max_num_evaluations: int = 128,
    patience: int = 64,
    stride: int = 4,
    loss_function: Callable = mean_squared_error,
    display_images: bool = False,
    **other_fixed_parameters,
):
    """
    Calibrates denoiser using self-supervised loss from Batson & Royer*
    Derived from code here:
    https://scikit-image.org/docs/dev/auto_examples/filters/plot_j_invariant_tutorial.html
    Reference: "Noise2Self: Blind Denoising by Self-Supervision, International
    Conference on Machine Learning, p. 524-533 (2019)"

    This 'smart' version uses a fancy optimizer to try to minimise the number
    of evaluations needed.

    Parameters
    ----------
    image: ArrayLike
        Image to calibate denoiser with.
    denoise_function: Callable
        Denosing function to calibrate. Should take an image as first parameter,
        all other parameters should have defaults
    denoise_parameters:
        Dictionary with keys corresponding to parameters of the denoising function.
        Values are either: (i) a list of possible values (categorical parameter),
        or (ii) a tuple of floats defining the bounds of that numerical parameter.
    max_num_evaluations: int
        Max number of function evaluations. This is per the size of the cartesian
        product of categorical parameters.
    patience : int
        After 'patience' evaluations we stop the optimiser
    stride: int
        Stride to compute self-supervised loss.
    loss_function: Callable
        Loss/Error function: takes two arrays and returns a distance-like function.
        Can be:  structural_error, mean_squared_error, _mean_absolute_error
    display_images: bool
        If True the denoised images for each parameter tested are displayed.
        this _will_ be slow.
    other_fixed_parameters: dict
        Other fixed parameters to pass to the denoiser function.


    Returns
    -------
    Dictionary with optimal parameters

    """

    # Pass fixed parameters:
    denoise_function = partial(denoise_function, **other_fixed_parameters)

    # for display purposes:
    denoised_images = []

    with lsection("Calibrating denoiser:"):

        # Generate mask:
        mask = _generate_mask(image, stride)

        # first we separate the categorical from numerical parameters;
        categorical_parameters = {}
        numerical_parameters = {}
        for name, value in denoise_parameters.items():
            if type(value) is list:
                # This is a categorical parameter:
                categorical_parameters[name] = value
            elif type(value) is tuple:
                # This is a numerical parameter:
                numerical_parameters[name] = value

        # Let's fix the order of numerical parameters:
        numerical_parameters_names = list(numerical_parameters.keys())
        numerical_parameters_bounds = list(
            numerical_parameters[n] for n in numerical_parameters_names
        )
        lprint(f"Parameter names : {numerical_parameters_names}")
        lprint(f"Parameter bounds: {numerical_parameters_bounds}")

        # number of numerical parameters:
        n = len(numerical_parameters_names)

        # Lets' expand the categorical parameters into their cartesian product:
        expanded_categorical_parameters = list(
            _product_from_dict(categorical_parameters)
        )

        # We remember the best combination, parameters and values:
        best_combination = None
        best_numerical_parameters = None
        best_loss_value = -math.inf

        # we optimise for each such combination of categorical parameters:
        with lsection(f"Going through {n} combinations of categorical parameters:"):

            for combination in expanded_categorical_parameters:
                with lsection(
                    f"Optimising categorical parameters combination: {combination}"
                ):

                    # This is the fucntion to optimize:
                    def opt_function(*point):

                        # Given a point, we build the parameter dict:
                        param = dict(combination)
                        for name, value in zip(numerical_parameters_names, point):
                            param[name] = value

                        # We compute the J-inv loss:
                        loss = _j_invariant_loss(
                            image,
                            denoise_function,
                            mask=mask,
                            loss_function=loss_function,
                            denoiser_kwargs=param,
                        )

                        if math.isnan(loss) or math.isinf(loss):
                            loss = math.inf

                        # lprint(f"({point}) -> {loss}")
                        # denoise image and store it, if needed:
                        if display_images:
                            denoised_image = denoise_function(image, **param)
                            denoised_images.append(denoised_image)

                        # our optimizer is a maximiser!
                        return -loss

                    # We optimise here:
                    new_parameters, new_loss_value = Optimizer().optimize(
                        opt_function,
                        bounds=numerical_parameters_bounds,
                        max_num_evaluations=max_num_evaluations,
                        patience=patience,
                    )

                    # We check if this is, or not, the best combination to date:
                    if (
                        new_loss_value > best_loss_value
                    ):  # our optimizer is a maximiser!
                        best_combination = combination
                        best_loss_value = new_loss_value
                        best_numerical_parameters = new_parameters

                    lprint(
                        f"Best numerical parameters: {best_numerical_parameters} for combination: {combination}"
                    )

            # Let convert back the parameters into a dictionary:
            best_parameters = best_combination | {
                name: float(value)
                for name, value in zip(
                    numerical_parameters_names, best_numerical_parameters
                )
            }

            lprint(f"Best parameters: {best_parameters}")

    # Display if needed:
    if display_images:
        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(numpy.stack(denoised_images), name='denoised')

    return best_parameters | other_fixed_parameters
