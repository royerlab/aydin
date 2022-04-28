import importlib

from aydin import Classic
from aydin.restoration import denoise
from aydin.restoration.denoise.base import DenoiseRestorationBase
from aydin.restoration.denoise.noise2selfcnn import Noise2SelfCNN
from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR


def get_pretrained_denoiser_class_instance(loaded_model_it):
    if "Classic" in loaded_model_it.__class__.__name__:
        denoiser_class = Classic
    elif "FGR" in loaded_model_it.__class__.__name__:
        denoiser_class = Noise2SelfFGR
    elif "CNN" in loaded_model_it.__class__.__name__:
        denoiser_class = Noise2SelfCNN
    else:
        raise ValueError(
            "Loaded model is not supported on restoration level implementations."
        )

    denoiser = denoiser_class()
    denoiser.it = loaded_model_it

    return denoiser


def get_denoiser_class_instance(variant, lower_level_args=None, it_transforms=None):
    """Returns instance of denoiser given a variant name, a set of arguments,
    and a list of transforms.

    Parameters
    ----------
    variant
        Variant to use
    lower_level_args
        Lower level args
    it_transforms
        Transforms to use.

    Returns
    -------
    Corresponding Restoration class
    """
    method_name_and_approach, implementation_name = variant.split("-")
    response = importlib.import_module(
        denoise.__name__ + '.' + method_name_and_approach.lower()
    )

    elem = [x for x in dir(response) if method_name_and_approach.lower() in x.lower()][
        0
    ]  # class name

    denoiser_class = response.__getattribute__(elem)

    if (
        it_transforms is None
        and lower_level_args is not None
        and lower_level_args["processing"]
    ):
        it_transforms = lower_level_args["processing"]

    return denoiser_class(
        lower_level_args=lower_level_args, it_transforms=it_transforms
    )


def get_list_of_denoiser_implementations():
    """
    Returns a list of denoiser implementations.
    Returns
    -------
    List of denoiser implementations.
    """
    denoiser_implementations = []
    descriptions = []

    for module in DenoiseRestorationBase.get_implementations_in_a_module(denoise):
        response = importlib.import_module(denoise.__name__ + '.' + module.name)

        elem = [x for x in dir(response) if module.name.replace('_', '') in x.lower()][
            0
        ]  # class name

        denoiser_class = response.__getattribute__(elem)
        denoiser_implementations += denoiser_class().implementations
        descriptions += denoiser_class().implementations_description

    return denoiser_implementations, descriptions
