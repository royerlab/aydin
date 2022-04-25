import math
from functools import partial
from typing import Callable, Dict, Union, Tuple, List, Any, Optional
import numpy
from scipy.optimize import minimize, shgo

from aydin.util.j_invariance.losses import mean_squared_error, mean_absolute_error
from aydin.util.j_invariance.util import (
    _generate_mask,
    _product_from_dict,
    _j_invariant_loss,
    _interpolate_image,
)
from aydin.util.log.log import lsection, lprint
from aydin.util.optimizer.optimizer import Optimizer


def calibrate_denoiser(
    image,
    denoise_function: Callable,
    denoise_parameters: Dict[str, Union[Tuple[float, float], List[Any]]],
    mode: str = 'smart',
    max_num_evaluations: int = 128,
    patience: int = 64,
    interpolation_mode: str = 'gaussian',
    stride: int = 4,
    loss_function: str = 'L1',
    blind_spots: Optional[List[Tuple[int]]] = None,
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

    mode: str
        Algorithm to use. Can be 'smart' and 'l-bfgs-b'.

    max_num_evaluations: int
        Max number of function evaluations. This is per the size of the cartesian
        product of categorical parameters.

    patience : int
        After 'patience' evaluations we stop the optimiser

    interpolation_mode: str
        When masking we need to use a value for replacing the masked values.
        One approach is to replace by zero: 'zero', or apply a Gaussian
        filter: 'gaussian', or apply a median filter: 'median'

    stride: int
        Stride to compute self-supervised loss.

    loss_function: str
        Loss/Error function: Can be:  'L1', 'L2', 'SSIM'

    blind_spots: Optional[List[Tuple[int]]]
        Set to None to enable automatic blind-spot detection.
        Otherwise provide list of blindspots. For example:
        [(0,0,0),(1,0,0),(-1,0,0)].

    display_images: bool
        If True the denoised images for each parameter tested are displayed.
        this _will_ be slow.

    other_fixed_parameters: dict
        Other fixed parameters to pass to the denoiser function.


    Returns
    -------
    Dictionary with optimal parameters

    """

    # Loss function:
    if loss_function == 'L1':
        loss_function = mean_absolute_error
    elif loss_function == 'L2':
        loss_function = mean_squared_error
    elif loss_function == 'SSIM':
        loss_function = mean_squared_error
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")

    # Pass fixed parameters:
    denoise_function = partial(denoise_function, **other_fixed_parameters)

    # for display purposes:
    denoised_images = []

    with lsection("Calibrating denoiser:"):

        # Convert image to float if that is not already the case:
        image = image.astype(dtype=numpy.float32, copy=False)

        # Generate mask:
        mask = _generate_mask(image, stride, blind_spots=blind_spots)

        # Compute interpolated image:
        interpolation = _interpolate_image(
            image, mask, num_iterations=2 * stride, mode=interpolation_mode
        )

        # Masked input image (fixed during optimisation!):
        masked_input_image = interpolation.copy()

        # We backup the masked input image to make sure that it is unchanged after optimisation,
        # which would indicate a 'non-behaving' denoiser that ioperates 'in-place'...
        masked_input_image_backup = masked_input_image.copy()

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

                    # This is the function to optimize:
                    def opt_function(*point):

                        # Given a point, we build the parameter dict:
                        param = dict(combination)
                        for name, value in zip(numerical_parameters_names, point):
                            param[name] = value

                        # We compute the J-inv loss:
                        loss = _j_invariant_loss(
                            image,
                            masked_input_image,
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

                    # First we check if there is no numerical parameters to optimise:
                    if len(numerical_parameters_bounds) == 0:
                        # If there is no numerical parameters to optimise, we just get the value for that categorical combination:
                        new_parameters = {}
                        new_loss_value = opt_function(new_parameters)
                        lprint(f"Loss: {new_loss_value}")
                    else:

                        if mode == "smart":
                            with lsection(
                                f"Searching by 'smart optimiser' the best denoising parameters among: {numerical_parameters}"
                            ):

                                # initialise optimiser:
                                optimiser: Optimizer = Optimizer()

                                # We optimise here:
                                new_parameters, new_loss_value = optimiser.optimize(
                                    opt_function,
                                    bounds=numerical_parameters_bounds,
                                    max_num_evaluations=max_num_evaluations,
                                    patience=patience,
                                )

                                pass

                        elif mode == "fast":
                            with lsection(
                                f"Searching by SHGO followed by L-BFGS-B the best denoising parameters among: {numerical_parameters}"
                            ):

                                def callback(x):
                                    lprint(x)

                                # x0 = numpy.asarray(tuple((0.5 * (v[1] - v[0]) for (n, v) in
                                #            numerical_parameters.items())))
                                bounds = list(
                                    [v[0:2] for (n, v) in numerical_parameters.items()]
                                )

                                # Impedance mismatch:
                                def __function(_denoiser_args):
                                    return -opt_function(*tuple(_denoiser_args))

                                # First we optimise with a global optimiser:
                                result = shgo(
                                    func=__function,
                                    bounds=bounds,
                                    sampling_method='sobol',
                                    options={
                                        'maxev': max_num_evaluations // 4,
                                        'disp': False,
                                    },
                                    callback=callback,
                                )
                                lprint(f"Global optimisation success: {result.success}")
                                lprint(
                                    f"Global optimisation convergence message: {result.message}"
                                )
                                lprint(
                                    f"Global optimisation number of function evaluations: {result.nfev}"
                                )
                                lprint(
                                    f"Best parameters until now: {result.x} for loss: {result.fun}"
                                )

                                # starting point for next ioptimkisation round is result of previous step:
                                x0 = result.x

                                # local optimisation using L-BFGS-B:
                                gtol = 1e-12
                                eps = 1e-10
                                for i in range(4):
                                    result = minimize(
                                        fun=__function,
                                        x0=x0,
                                        method='L-BFGS-B',
                                        bounds=bounds,
                                        options={
                                            'maxfun': max_num_evaluations,
                                            'eps': eps,
                                            'ftol': 1e-8,
                                            'gtol': gtol,
                                        },
                                        callback=callback,
                                    )
                                    if (
                                        'NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
                                        not in result.message
                                    ):
                                        break
                                    x0 = result.x
                                    eps *= 100

                                lprint(f"Local optimisation success: {result.success}")
                                lprint(
                                    f"Local optimisation convergence message: {result.message}"
                                )
                                lprint(
                                    f"Local optimisation number of function evaluations: {result.nfev}"
                                )
                                lprint(
                                    f"Best parameters until now: {result.x} for loss: {result.fun}"
                                )

                                # We optimise here:
                                new_parameters, new_loss_value = result.x, result.fun

                        else:
                            raise ValueError(f"Unknown optimisation mode: {mode}")

                    # We check if this is, or not, the best combination to date:
                    if (
                        new_loss_value > best_loss_value
                    ):  # our optimizer is a maximiser!

                        best_numerical_parameters = new_parameters
                        best_combination = combination
                        best_loss_value = new_loss_value

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

        # We check that the masked input image is unchanged:
        if not numpy.array_equal(masked_input_image_backup, masked_input_image):
            raise RuntimeError(
                "The denoiser being calibrated is modifying the input image! Calibration will not be accurate!"
            )

    # Display if needed:
    if display_images:
        try:
            import napari

            viewer = napari.Viewer()
            viewer.add_image(image, name='image')
            viewer.add_image(numpy.stack(denoised_images), name='denoised')
            napari.run()
        except Exception:
            lprint(
                "Problem while trying to display images obtained during optimization"
            )

    return best_parameters | other_fixed_parameters
