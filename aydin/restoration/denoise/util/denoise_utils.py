"""Utility functions for discovering and instantiating denoisers.

Provides helper functions to create denoiser instances from variant names,
load pre-trained models, and enumerate all available denoiser implementations.
"""

import importlib

from aydin import Classic
from aydin.restoration import denoise
from aydin.restoration.denoise.base import DenoiseRestorationBase
from aydin.util.log.log import aprint


def get_pretrained_denoiser_class_instance(loaded_model_it):
    """Create a denoiser restoration instance from a loaded image translator.

    Inspects the class name of the loaded model to determine the correct
    denoiser wrapper (Classic, Noise2SelfFGR, or Noise2SelfCNN).

    Parameters
    ----------
    loaded_model_it : ImageTranslatorBase
        A loaded image translator instance.

    Returns
    -------
    DenoiseRestorationBase
        The corresponding denoiser wrapper with ``it`` set to the loaded model.

    Raises
    ------
    ValueError
        If the loaded model type is not recognised.
    """
    if "Classic" in loaded_model_it.__class__.__name__:
        denoiser_class = Classic
    elif "FGR" in loaded_model_it.__class__.__name__:
        from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR

        denoiser_class = Noise2SelfFGR
    elif "CNN" in loaded_model_it.__class__.__name__:
        from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN

        denoiser_class = Noise2SelfCNN
    else:
        raise ValueError(
            "Loaded model is not supported on restoration level implementations."
        )

    denoiser = denoiser_class()
    denoiser.it = loaded_model_it

    return denoiser


def get_denoiser_class_instance(variant, lower_level_args=None, it_transforms=None):
    """Create a denoiser instance from a variant string.

    The variant string has the format ``'MethodName-implementation'``
    (e.g. ``'Classic-butterworth'``, ``'Noise2SelfFGR-cb'``).

    Parameters
    ----------
    variant : str
        Full variant identifier in the format ``'Method-implementation'``.
    lower_level_args : dict, optional
        Additional low-level arguments passed to the denoiser constructor.
        If it contains a ``'processing'`` key and ``it_transforms`` is
        ``None``, the processing value is used as the transform list.
    it_transforms : list of dict, optional
        Custom list of image transforms to apply.

    Returns
    -------
    DenoiseRestorationBase
        An instance of the appropriate denoiser class configured with
        the specified variant and arguments.
    """
    method_name_and_approach, implementation_name = variant.split("-", 1)
    response = importlib.import_module(
        denoise.__name__ + '.' + method_name_and_approach.lower()
    )

    candidates = [
        x for x in dir(response) if method_name_and_approach.lower() in x.lower()
    ]
    if not candidates:
        raise ValueError(
            f"No denoiser class found for method '{method_name_and_approach}'"
        )
    elem = candidates[0]  # class name

    denoiser_class = response.__getattribute__(elem)

    if (
        it_transforms is None
        and lower_level_args is not None
        and lower_level_args["processing"]
    ):
        it_transforms = lower_level_args["processing"]

    return denoiser_class(
        variant=implementation_name,
        lower_level_args=lower_level_args,
        it_transforms=it_transforms,
    )


def get_list_of_denoiser_implementations():
    """Discover all available denoiser implementations and their descriptions.

    Iterates over all denoiser modules (Classic, Noise2SelfFGR, Noise2SelfCNN)
    and collects their implementation variants.

    Returns
    -------
    tuple of (list of str, list of str)
        A tuple containing the list of implementation variant names and their
        corresponding human-readable descriptions.
    """
    denoiser_implementations = []
    descriptions = []

    for module in DenoiseRestorationBase.get_implementations_in_a_module(denoise):
        try:
            response = importlib.import_module(denoise.__name__ + '.' + module.name)

            candidates = [
                x for x in dir(response) if module.name.replace('_', '') in x.lower()
            ]
            if not candidates:
                continue
            elem = candidates[0]  # class name

            denoiser_class = response.__getattribute__(elem)
            denoiser_implementations += denoiser_class().implementations
            descriptions += denoiser_class().implementations_description
        except Exception as e:
            aprint(
                f"Warning: Denoiser '{module.name}' failed to load: {e}. "
                "Check that its dependencies are installed."
            )
            continue

    return denoiser_implementations, descriptions
