import os
import shutil
import sys
from glob import glob
import click
import numpy
import napari

from aydin.gui.gui import run
from aydin.it.base import ImageTranslatorBase
from aydin.restoration.deconvolve.lr import LucyRichardson
from aydin.io.io import imwrite, imread
from aydin.io.utils import get_output_image_path
from aydin.restoration.denoise.util.denoise_utils import get_denoiser_class_instance
from aydin.util.misc.json import load_any_json
from aydin.util.log.log import lprint, Log
from aydin.util.misc.slicing_helper import apply_slicing
from aydin import __version__

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

VERSION = __version__


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.version_option(version=VERSION)
def cli(ctx):
    """aydin cli

    Parameters
    ----------
    ctx : click.Context

    """
    sys._excepthook = sys.excepthook

    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)

    sys.excepthook = exception_hook

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To disable TensorFlow logs
    if ctx.invoked_subcommand is None:
        run(VERSION)
    else:
        Log.enable_output = True
        pass


@cli.command()
@click.argument('files', nargs=-1)
@click.option('-ts', '--training-slicing', default='', type=str)
@click.option('-is', '--inference-slicing', default='', type=str)
@click.option('-v', '--variant', default="noise2selffgr-cb")
@click.option('--use-model/--save-model', default=False)
@click.option('--model-path', default=None)
@click.option('--lower-level-args', default=None)
def denoise(files, **kwargs):
    """denoise command

    Parameters
    ----------
    files
    kwargs : dict

    """
    # Check whether a path is provided for a model to use or save
    input_model_path = kwargs["model_path"] if kwargs["model_path"] else None

    # Check whether a filename is provided for lower-level-args json
    if kwargs["lower_level_args"]:
        lower_level_args = load_any_json(kwargs['lower_level_args'])
        backend = lower_level_args["variant"]
    else:
        lower_level_args = None
        backend = kwargs["variant"]

    filenames = []
    for filename in files:
        # if our shell does not do filename globbing
        expanded = list(glob(filename))

        if len(expanded) == 0 and '*' not in filename:
            raise (click.BadParameter('{}: file not found'.format(filename)))
        filenames.extend(expanded)

    for filename in filenames:

        # Get abspath to image and read it
        path = os.path.abspath(filename)
        noisy, noisy_metadata = imread(path)

        noisy2train = apply_slicing(noisy, kwargs['training_slicing'])
        noisy2infer = apply_slicing(noisy, kwargs['inference_slicing'])

        if kwargs['use_model']:
            shutil.unpack_archive(
                input_model_path, os.path.dirname(input_model_path), "zip"
            )
            it = ImageTranslatorBase.load(input_model_path[:-4])

            # Predict the resulting image
            response = it.translate(
                noisy2infer,
                batch_axes=noisy_metadata.batch_axes,
                channel_axes=noisy_metadata.channel_axes,
                tile_size=kwargs['tile_size'] if 'tile_size' in kwargs else None,
            )

            denoised = response.astype(noisy2infer.dtype, copy=False)
            shutil.rmtree(input_model_path[:-4])
        else:
            n2s = get_denoiser_class_instance(
                lower_level_args=lower_level_args, variant=backend
            )

            n2s.train(
                noisy2train,
                batch_axes=noisy_metadata.batch_axes
                if noisy_metadata is not None
                else None,
                chan_axes=noisy_metadata.channel_axes
                if noisy_metadata is not None
                else None,
                image_path=path,
                **kwargs,
            )

            denoised = n2s.denoise(
                noisy2infer,
                batch_axes=noisy_metadata.batch_axes
                if noisy_metadata is not None
                else None,
                chan_axes=noisy_metadata.channel_axes
                if noisy_metadata is not None
                else None,
            )

        path, index_counter = get_output_image_path(path)
        imwrite(denoised, path)
        lprint("DONE")


@cli.command()
@click.argument('files', nargs=-1)
@click.argument('psf_path', nargs=1)
@click.option('-s', '--slicing', default='', type=str)
@click.option('-b', '--backend', default=None)
def lucyrichardson(files, psf_path, **kwargs):
    """lucyrichardson command

    Parameters
    ----------
    files
    psf_kernel
    kwargs : dict

    """
    psf_kernel = None
    if psf_path != "":
        psf_kernel = imread(psf_path, value_norm=False)[0]
        psf_kernel = psf_kernel.astype(numpy.float32, copy=False)
        psf_kernel /= psf_kernel.sum()

    filepaths, image_arrays, metadatas = handle_files(files, kwargs['slicing'])
    for filepath, input_image, metadata in zip(filepaths, image_arrays, metadatas):
        lr = LucyRichardson(
            psf_kernel=psf_kernel, max_num_iterations=20, backend=kwargs['backend']
        )

        lr.train(input_image, input_image)
        deconvolved = lr.deconvolve(input_image)

        path, index_counter = get_output_image_path(filepath, "deconvolved")
        imwrite(deconvolved, path)


@cli.command()
@click.argument('files', nargs=-1)
@click.option('-s', '--slicing', default='', type=str)
def info(files, **kwargs):
    """aydin info command

    Parameters
    ----------
    files
    kwargs

    """
    handle_files(
        files, kwargs['slicing']
    )  # we are not printing anything here as aydin.io.imread prints logs


@cli.command()
@click.argument('files', nargs=-1)
@click.option('-s', '--slicing', default='', type=str)
def view(files, **kwargs):
    """aydin view command

    Parameters
    ----------
    files
    kwargs

    Returns
    -------

    """
    filenames, image_arrays, metadatas = handle_files(files, kwargs['slicing'])

    with napari.gui_qt():
        viewer = napari.Viewer()

        for idx, image in enumerate(image_arrays):
            viewer.add_image(image, name=filenames[idx])


@cli.command()
@click.argument('files', nargs=-1)
@click.option('-s', '--slicing', default='', type=str)
def hyperstack(files, **kwargs):
    """aydin hyperstack command

    Parameters
    ----------
    files
    kwargs : dict

    """
    filenames, image_arrays, metadatas = handle_files(files, kwargs['slicing'])
    result_path = filenames[0]

    stacked_image = numpy.stack(image_arrays)

    result_path, index_counter = get_output_image_path(result_path)
    imwrite(stacked_image, result_path)


def handle_files(files, slicing):
    """Handle files

    Parameters
    ----------
    files
    slicing

    Returns
    -------

    """
    filepaths = []
    image_arrays = []
    metadatas = []

    for filename in files:
        # if our shell does not do filename globbing
        expanded = list(glob(filename))
        if len(expanded) == 0 and '*' not in filename:
            raise (click.BadParameter('{}: file not found'.format(filename)))
        filepaths.extend(expanded)

    for idx, filename in enumerate(filepaths):
        # Get abspath to image and read it
        path = os.path.abspath(filename)

        input_image, input_metadata = imread(path)
        input_image = apply_slicing(input_image, slicing)
        image_arrays.append(input_image)
        metadatas.append(input_metadata)

    return filepaths, image_arrays, metadatas


if __name__ == '__main__':
    cli()
