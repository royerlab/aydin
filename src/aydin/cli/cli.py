"""Command-line interface for Aydin image denoising.

This module defines the Click-based CLI for Aydin, providing commands for
image denoising, viewing, analysis (SSIM, PSNR, MSE, FSC), channel splitting,
hyperstacking, and benchmarking of denoising algorithms.

The CLI is built with `Click <https://click.palletsprojects.com/>`_ and
registered as the ``aydin`` console script entry point.  When invoked without
a subcommand the Aydin Studio GUI is launched; otherwise the specified
subcommand is dispatched.

Attributes
----------
CONTEXT_SETTINGS : dict
    Click context settings enabling ``-h`` as a help shortcut.
VERSION : str
    Current Aydin version string, imported from ``aydin.__version__``.
"""

import os
import sys
from glob import glob

import click

from aydin import __version__
from aydin.cli.styling import (
    AydinGroup,
    format_cite_box,
    format_denoiser_listing,
    styled_banner,
    styled_metric,
    success_message,
)

CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help'],
    max_content_width=88,
)

VERSION = __version__


@click.group(
    cls=AydinGroup,
    invoke_without_command=True,
    context_settings=CONTEXT_SETTINGS,
)
@click.pass_context
@click.version_option(version=VERSION)
def cli(ctx):
    """Launch Aydin Studio GUI or invoke a subcommand.

    When called without a subcommand, launches the Aydin Studio GUI.
    Otherwise, dispatches to the specified subcommand.
    \f

    Parameters
    ----------
    ctx : click.Context
        Click context object for subcommand dispatch.
    """
    sys._excepthook = sys.excepthook

    def exception_hook(exctype, value, traceback):
        """Log unhandled exceptions and exit with a non-zero status code.

        Parameters
        ----------
        exctype : type
            The exception class.
        value : BaseException
            The exception instance.
        traceback : types.TracebackType
            The traceback object.
        """
        from aydin.util.log.log import aprint

        aprint(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)

    sys.excepthook = exception_hook

    from aydin.util.log.log import Log

    Log.enable_output = True

    if ctx.invoked_subcommand is None:
        click.echo(styled_banner())
        click.echo()
        click.echo(click.style('  Loading Aydin Studio...', dim=True))

        from aydin.gui.gui import run

        click.echo(success_message('Aydin Studio ready'))
        click.echo()
        run(VERSION)


def _list_denoisers_callback(ctx, param, value):
    """Print available denoiser variants and exit."""
    if not value or ctx.resilient_parsing:
        return
    from aydin.restoration.denoise.util.denoise_utils import (
        get_list_of_denoiser_implementations,
    )

    names, descriptions = get_list_of_denoiser_implementations()
    click.echo(format_denoiser_listing(names, descriptions))
    ctx.exit()


def _expand_file_globs(files):
    """Expand glob patterns and validate that literal paths exist.

    Handles shells that do not perform filename globbing by expanding
    wildcards internally.  Literal (non-glob) paths that do not match
    any existing file raise a :class:`click.BadParameter` error.

    Parameters
    ----------
    files : tuple of str
        File paths or glob patterns.

    Returns
    -------
    list of str
        Expanded list of matching file paths.

    Raises
    ------
    click.BadParameter
        If a literal file path does not match any existing file.
    """
    expanded = []
    for filename in files:
        matches = list(glob(filename))
        if not matches and '*' not in filename:
            raise click.BadParameter(f'{filename}: file not found')
        expanded.extend(matches)
    return expanded


@cli.command()
@click.argument('files', nargs=-1)
@click.option(
    '-ts',
    '--training-slicing',
    default='',
    type=str,
    metavar='SLICE',
    help='NumPy-style slicing for training (e.g. "[0:100,:]").',
)
@click.option(
    '-is',
    '--inference-slicing',
    default='',
    type=str,
    metavar='SLICE',
    help='NumPy-style slicing for inference.',
)
@click.option(
    '-ba',
    '--batch-axes',
    type=str,
    help='Axes to treat as batch dimensions (single-image only).',
)
@click.option(
    '-ca',
    '--channel-axes',
    type=str,
    help='Axes to treat as channel dimensions (single-image only).',
)
@click.option(
    '-d',
    '--denoiser',
    'denoiser_name',
    default='noise2selffgr-cb',
    metavar='VARIANT',
    help='Denoiser variant. Use --list-denoisers to see all.',
)
@click.option(
    '--use-model/--save-model',
    default=False,
    help='Load a pre-trained model instead of training a new one.',
)
@click.option(
    '--model-path',
    default=None,
    metavar='PATH',
    help='Path to a saved model .zip archive (required with --use-model).',
)
@click.option(
    '--lower-level-args',
    default=None,
    metavar='JSON',
    help='Path to a JSON file with lower-level denoiser configuration.',
)
@click.option(
    '--output-folder',
    default='',
    metavar='DIR',
    help='Directory for output files (default: same as source image).',
)
@click.option(
    '--list-denoisers',
    is_flag=True,
    callback=_list_denoisers_callback,
    expose_value=False,
    is_eager=True,
    help='List available denoiser variants and exit.',
)
def denoise(
    files,
    training_slicing,
    inference_slicing,
    batch_axes,
    channel_axes,
    denoiser_name,
    use_model,
    model_path,
    lower_level_args,
    output_folder,
):
    """Denoise one or more image files.

    Reads each input image, trains a denoiser (or loads a pre-trained model),
    applies denoising, and writes the result to disk.  When --use-model is
    specified, a previously saved model archive is loaded instead of training a
    new one.  Otherwise a fresh denoiser is trained and the model is saved
    alongside the output image.
    \f

    Parameters
    ----------
    files : tuple of str
        Input image file paths or glob patterns.
    training_slicing : str
        NumPy-style slicing string applied to the image before training.
    inference_slicing : str
        NumPy-style slicing string applied to the image before inference.
    batch_axes : str or None
        Axes to treat as batch dimensions (only for single-image denoising).
    channel_axes : str or None
        Axes to treat as channel dimensions (only for single-image denoising).
    denoiser_name : str
        Denoiser variant identifier (default ``'noise2selffgr-cb'``).
    use_model : bool
        If ``True``, load a pre-trained model from *model_path*.
    model_path : str or None
        Path to a saved model archive (``.zip``).
    lower_level_args : str or None
        Path to a JSON file with lower-level denoiser configuration.
    output_folder : str
        Directory for output files; defaults to the source image directory.

    Raises
    ------
    click.BadParameter
        If a specified file path does not exist and is not a glob pattern.
    """
    import ast
    import shutil
    import tempfile

    from aydin.io.io import imread, imwrite
    from aydin.io.utils import get_output_image_path, get_save_model_path
    from aydin.restoration.denoise.util.denoise_utils import get_denoiser_class_instance
    from aydin.util.misc.json import load_any_json
    from aydin.util.misc.slicing_helper import apply_slicing

    # Validate --use-model requires --model-path
    if use_model and model_path is None:
        raise click.BadParameter('--model-path is required when using --use-model')

    # Check whether a filename is provided for lower-level-args json
    if lower_level_args:
        lower_level_args = load_any_json(lower_level_args)
        variant = lower_level_args["variant"]
    else:
        lower_level_args = None
        variant = denoiser_name

    filenames = _expand_file_globs(files)

    for filename in filenames:

        # Get abspath to image and read it
        path = os.path.abspath(filename)
        noisy, noisy_metadata = imread(path)

        noisy2train = apply_slicing(noisy, training_slicing)
        noisy2infer = apply_slicing(noisy, inference_slicing)

        if batch_axes is not None and len(filenames) == 1:
            noisy_metadata.batch_axes = ast.literal_eval(batch_axes)

        if channel_axes is not None and len(filenames) == 1:
            noisy_metadata.channel_axes = ast.literal_eval(channel_axes)

        out_path, index_counter = get_output_image_path(
            path, operation_type="denoised", output_folder=output_folder
        )

        if use_model:
            _model_root, model_ext = os.path.splitext(model_path)
            if model_ext.lower() != '.zip':
                raise click.BadParameter(
                    f"Model path must be a .zip archive, got '{model_ext}'"
                )

            from aydin.it.base import ImageTranslatorBase

            with tempfile.TemporaryDirectory() as tmpdir:
                shutil.unpack_archive(model_path, tmpdir, "zip")
                model_dirs = [
                    d
                    for d in os.listdir(tmpdir)
                    if os.path.isdir(os.path.join(tmpdir, d))
                ]
                if not model_dirs:
                    raise click.BadParameter(
                        f"Model archive '{model_path}' contains no model directory"
                    )
                model_root = os.path.join(tmpdir, model_dirs[0])
                it = ImageTranslatorBase.load(model_root)

                # Predict the resulting image
                response = it.translate(
                    noisy2infer,
                    batch_axes=noisy_metadata.batch_axes,
                    channel_axes=noisy_metadata.channel_axes,
                )

                denoised = response.astype(noisy2infer.dtype, copy=False)
        else:
            denoiser_instance = get_denoiser_class_instance(
                lower_level_args=lower_level_args, variant=variant
            )

            denoiser_instance.train(
                noisy2train,
                batch_axes=(
                    noisy_metadata.batch_axes if noisy_metadata is not None else None
                ),
                channel_axes=(
                    noisy_metadata.channel_axes if noisy_metadata is not None else None
                ),
            )

            denoised = denoiser_instance.denoise(
                noisy2infer,
                batch_axes=(
                    noisy_metadata.batch_axes if noisy_metadata is not None else None
                ),
                channel_axes=(
                    noisy_metadata.channel_axes if noisy_metadata is not None else None
                ),
            )

            save_path = get_save_model_path(
                path,
                passed_counter=index_counter,
                output_folder=output_folder,
            )
            denoiser_instance.save(save_path)

        imwrite(denoised, out_path)
        click.echo(success_message("Denoising complete"))


@cli.command()
@click.argument('files', nargs=-1)
@click.option(
    '-s',
    '--slicing',
    default='',
    type=str,
    metavar='SLICE',
    help='NumPy-style slicing applied to each image.',
)
def info(files, slicing):
    """Display metadata and information about image files.

    Reads each input file and prints its metadata (shape, dtype, axes, etc.)
    to standard output via the Aydin logging system.
    \f

    Parameters
    ----------
    files : tuple of str
        Input image file paths or glob patterns.
    slicing : str
        NumPy-style slicing specification string applied to each image.

    Raises
    ------
    click.BadParameter
        If a specified file path does not exist and is not a glob pattern.
    """
    handle_files(
        files, slicing
    )  # we are not printing anything here as aydin.io.imread prints logs


@cli.command()
@click.argument('files', nargs=-1)
@click.option(
    '-s',
    '--slicing',
    default='',
    type=str,
    metavar='SLICE',
    help='NumPy-style slicing applied to each image.',
)
def view(files, slicing):
    """View image files in a napari viewer.

    Reads each input file and displays it as a separate layer in an
    interactive napari viewer window.  The viewer blocks until the user
    closes the window.
    \f

    Parameters
    ----------
    files : tuple of str
        Input image file paths or glob patterns.
    slicing : str
        NumPy-style slicing specification string applied to each image.

    Raises
    ------
    click.BadParameter
        If a specified file path does not exist and is not a glob pattern.
    """
    filenames, image_arrays, _metadatas = handle_files(files, slicing)

    import napari

    viewer = napari.Viewer()

    for idx, image in enumerate(image_arrays):
        viewer.add_image(image, name=filenames[idx])

    napari.run()


@cli.command()
@click.argument('files', nargs=-1)
@click.option(
    '-s',
    '--slicing',
    default='',
    type=str,
    metavar='SLICE',
    help='NumPy-style slicing applied to each image.',
)
def split_channels(files, slicing):
    """Split multi-channel images into separate single-channel files.

    Reads each input image, splits along the channel axis defined in the
    image metadata, and writes each channel as a separate image file.  Output
    filenames are prefixed with channel_<idx>_.
    \f

    Parameters
    ----------
    files : tuple of str
        Input image file paths or glob patterns.
    slicing : str
        NumPy-style slicing specification string applied to each image.

    Raises
    ------
    click.BadParameter
        If a specified file path does not exist and is not a glob pattern.
    """
    from aydin.io.io import imwrite
    from aydin.io.utils import split_image_channels

    filenames, image_arrays, metadatas = handle_files(files, slicing)

    for filename, image_array, metadata in zip(filenames, image_arrays, metadatas):
        result = split_image_channels(image_array, metadata)
        if result is None:
            continue
        splitted_arrays, splitted_metadatas = result

        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        splitted_filenames = [
            os.path.join(dirname, f"channel_{i}_{basename}")
            for i in range(len(splitted_arrays))
        ]

        for splitted_filename, splitted_array, splitted_metadata in zip(
            splitted_filenames, splitted_arrays, splitted_metadatas
        ):
            imwrite(
                splitted_array, splitted_filename, splitted_metadata, overwrite=False
            )
            click.echo(success_message(f"Wrote {splitted_filename}"))


@cli.command()
@click.argument('files', nargs=-1)
@click.option(
    '-s',
    '--slicing',
    default='',
    type=str,
    metavar='SLICE',
    help='NumPy-style slicing applied to each image.',
)
def hyperstack(files, slicing):
    """Stack multiple images into a single higher-dimensional image.

    Reads all input images and stacks them along a new leading axis using
    numpy.stack, writing the result as a single TIFF file.  The output
    filename is derived from the first input file with a _hyperstacked suffix.
    \f

    Parameters
    ----------
    files : tuple of str
        Input image file paths or glob patterns.  All images must share the
        same shape and dtype for stacking to succeed.
    slicing : str
        NumPy-style slicing specification string applied to each image.

    Raises
    ------
    click.BadParameter
        If a specified file path does not exist and is not a glob pattern.
    ValueError
        If image shapes are incompatible for stacking.
    """
    import numpy

    from aydin.io.io import imwrite
    from aydin.io.utils import get_output_image_path

    filenames, image_arrays, _metadatas = handle_files(files, slicing)
    result_path = filenames[0]

    stacked_image = numpy.stack(image_arrays)

    result_path, _index_counter = get_output_image_path(
        result_path, operation_type="hyperstacked"
    )
    imwrite(stacked_image, result_path)
    click.echo(success_message(f"Wrote {result_path}"))


@cli.command()
@click.argument('files', nargs=2)
@click.option(
    '-s',
    '--slicing',
    default='',
    type=str,
    metavar='SLICE',
    help='NumPy-style slicing applied to each image.',
)
def ssim(files, slicing):
    """Compute the Structural Similarity Index (SSIM) between two images.

    Both images are normalised to the [0, 1] range and clipped before
    computing the SSIM. The result is printed to standard output.
    \f

    Parameters
    ----------
    files : tuple of str
        Exactly two image file paths.
    slicing : str
        NumPy-style slicing specification string applied to each image.

    Raises
    ------
    click.BadParameter
        If a specified file path does not exist and is not a glob pattern.
    """
    from skimage.metrics import structural_similarity

    from aydin.io.datasets import normalise
    from aydin.util.log.log import aprint

    _filenames, image_arrays, _metadatas = handle_files(files, slicing)

    value = structural_similarity(
        normalise(image_arrays[1]).clip(0, 1),
        normalise(image_arrays[0]).clip(0, 1),
        data_range=1.0,
    )
    aprint(styled_metric("ssim", value))


@cli.command()
@click.argument('files', nargs=2)
@click.option(
    '-s',
    '--slicing',
    default='',
    type=str,
    metavar='SLICE',
    help='NumPy-style slicing applied to each image.',
)
def psnr(files, slicing):
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Both images are normalised to the [0, 1] range and clipped before
    computing the PSNR. The first image is treated as the ground truth.
    \f

    Parameters
    ----------
    files : tuple of str
        Exactly two image file paths.
    slicing : str
        NumPy-style slicing specification string applied to each image.

    Raises
    ------
    click.BadParameter
        If a specified file path does not exist and is not a glob pattern.
    """
    from skimage.metrics import peak_signal_noise_ratio

    from aydin.io.datasets import normalise
    from aydin.util.log.log import aprint

    _filenames, image_arrays, _metadatas = handle_files(files, slicing)

    value = peak_signal_noise_ratio(
        normalise(image_arrays[1]).clip(0, 1),
        normalise(image_arrays[0]).clip(0, 1),
        data_range=1.0,
    )
    aprint(styled_metric("psnr", value))


@cli.command()
@click.argument('files', nargs=2)
@click.option(
    '-s',
    '--slicing',
    default='',
    type=str,
    metavar='SLICE',
    help='NumPy-style slicing applied to each image.',
)
def mse(files, slicing):
    """Compute the Mean Squared Error (MSE) between two images.

    Both images are normalised to the [0, 1] range and clipped before
    computing the MSE. The result is printed to standard output.
    \f

    Parameters
    ----------
    files : tuple of str
        Exactly two image file paths.
    slicing : str
        NumPy-style slicing specification string applied to each image.

    Raises
    ------
    click.BadParameter
        If a specified file path does not exist and is not a glob pattern.
    """
    from skimage.metrics import mean_squared_error

    from aydin.io.datasets import normalise
    from aydin.util.log.log import aprint

    _filenames, image_arrays, _metadatas = handle_files(files, slicing)

    value = mean_squared_error(
        normalise(image_arrays[1]).clip(0, 1),
        normalise(image_arrays[0]).clip(0, 1),
    )
    aprint(styled_metric("mse", value))


@cli.command()
@click.argument('files', nargs=2)
@click.option(
    '-s',
    '--slicing',
    default='',
    type=str,
    metavar='SLICE',
    help='NumPy-style slicing applied to each image.',
)
@click.option(
    '-o',
    '--output',
    default='fsc.png',
    type=str,
    metavar='PATH',
    help='Output file path for the FSC plot.',
)
def fsc(files, slicing, output):
    """Compute and plot the Fourier Shell Correlation (FSC) between two images.

    Both images are normalised before computing the FSC curve. The resulting
    correlation curve is plotted with matplotlib and saved to the output path
    (default fsc.png).
    \f

    Parameters
    ----------
    files : tuple of str
        Exactly two image file paths.
    slicing : str
        NumPy-style slicing string applied to each image before computation.
    output : str
        Output file path for the FSC plot (default ``fsc.png``).

    Raises
    ------
    click.BadParameter
        If a specified file path does not exist and is not a glob pattern.
    """
    from aydin.analysis.fsc import fsc as compute_fsc
    from aydin.io.datasets import normalise

    _filenames, image_arrays, _metadatas = handle_files(files, slicing)

    correlations = compute_fsc(normalise(image_arrays[0]), normalise(image_arrays[1]))

    import matplotlib.pyplot as plt

    plt.plot(correlations)
    plt.title('image1, image2')
    plt.savefig(output)
    plt.close()
    click.echo(success_message(f"Wrote {output}"))


def _snr_from_frequency(frequency, crop):
    """Estimate SNR in dB from pre-computed resolution frequency and crop.

    This mirrors the logic in ``aydin.analysis.snr_estimate.snr_estimate``
    but accepts the resolution frequency and crop directly so that
    ``resolution_estimate`` is not called a second time.

    Parameters
    ----------
    frequency : float
        Normalised cutoff frequency from ``resolution_estimate``.
    crop : numpy.ndarray
        Representative crop returned by ``resolution_estimate``.

    Returns
    -------
    float
        Estimated signal-to-noise ratio in dB.  Returns ``math.inf`` when
        no noise energy is detected above the frequency cutoff (e.g. the
        crop is constant or the denoiser removed all high-frequency content).
        Callers should interpret ``inf`` as "no measurable noise".
    """
    import math

    import numpy
    from numpy.linalg import norm
    from scipy.fft import dctn

    img = crop.astype(numpy.float32)
    img -= img.mean()
    variance = img.var()
    if variance == 0:
        # Constant crop — no signal or noise structure to measure.
        return math.inf
    img /= variance

    img_dct = dctn(img, workers=-1)

    # Frequency map
    f = numpy.zeros_like(img)
    axis_grid = tuple(numpy.linspace(0, 1, s) for s in img.shape)
    for x in numpy.meshgrid(*axis_grid, indexing='ij'):
        f += x**2
    f = numpy.sqrt(f)

    signal_domain = f <= frequency
    noise_domain = f > frequency

    signal_energy = norm(img_dct[signal_domain]) ** 2
    noise_energy = norm(img_dct[noise_domain]) ** 2

    signal_domain_volume = numpy.sum(signal_domain) / img_dct.size
    noise_domain_volume = 1 - signal_domain_volume

    corrected_noise_energy = noise_energy / max(noise_domain_volume, 1e-16)

    if corrected_noise_energy == 0:
        # No energy above the frequency cutoff — noise is unmeasurable.
        return math.inf

    corrected_signal_energy = (
        signal_energy - signal_domain_volume * corrected_noise_energy
    )
    corrected_signal_energy = max(1e-16, corrected_signal_energy)

    return 10 * math.log10(corrected_signal_energy / corrected_noise_energy)


@cli.command()
@click.argument('files', nargs=-1)
@click.option(
    '-s',
    '--slicing',
    default='',
    type=str,
    metavar='SLICE',
    help='NumPy-style slicing applied to each image.',
)
@click.option(
    '-nr',
    '--nbruns',
    default=1,
    type=int,
    help='Number of independent runs per denoiser.',
)
@click.option(
    '--save-denoised-images/--rm-denoised-images',
    default=False,
    help='Save each denoised output image to disk.',
)
def benchmark_algos(files, slicing, nbruns, save_denoised_images):
    """Benchmark all available denoising algorithms on the given images.

    Iterates over every registered denoiser variant and, for each input
    image, trains the denoiser, infers a denoised result, and evaluates
    three quality metrics: self-supervised loss, estimated SNR, and
    resolution estimate. Each metric is averaged over nbruns independent
    runs and the results are written to three CSV files in the current
    working directory.
    \f

    Parameters
    ----------
    files : tuple of str
        Input image file paths or glob patterns.
    slicing : str
        NumPy-style slicing specification string applied to each image.
    nbruns : int
        Number of independent training/inference runs per denoiser.
    save_denoised_images : bool
        If ``True``, save each denoised output image to disk.

    Raises
    ------
    click.BadParameter
        If a specified file path does not exist and is not a glob pattern.
    """
    import csv

    import numpy

    from aydin.analysis.resolution_estimate import resolution_estimate
    from aydin.io.datasets import normalise
    from aydin.io.io import imwrite
    from aydin.io.utils import get_output_image_path
    from aydin.restoration.denoise.util.denoise_utils import (
        get_denoiser_class_instance,
        get_list_of_denoiser_implementations,
    )
    from aydin.util.j_invariance.losses import mean_squared_error
    from aydin.util.j_invariance.util import _generate_mask, _interpolate_image
    from aydin.util.log.log import aprint

    filenames, image_arrays, metadatas = handle_files(files, slicing)

    denoiser_names, _ = get_list_of_denoiser_implementations()

    loss_function = mean_squared_error
    self_supervised_loss_results = {}
    estimated_snr_results = {}
    estimated_res_results = {}

    # Iterate over the input images
    for filename, image_array, _metadata in zip(filenames, image_arrays, metadatas):
        self_supervised_loss_results[filename] = {}
        estimated_snr_results[filename] = {}
        estimated_res_results[filename] = {}

        # Normalise and generate J-invariance mask
        input_image = normalise(image_array)
        mask = _generate_mask(input_image, stride=4, blind_spots=[])
        masked_input = _interpolate_image(
            input_image, mask, mode='gaussian', num_iterations=8
        )
        aprint(f"Image shape: {input_image.shape}, mask fraction: {mask.mean():.4f}")

        # Iterate over the available denoisers
        for denoiser_name in denoiser_names:
            ss_losses, snrs, res_estimates = [], [], []
            for run_idx in range(nbruns):
                try:
                    denoiser_instance = get_denoiser_class_instance(
                        variant=denoiser_name
                    )

                    # Train on original image (denoiser learns the signal)
                    denoiser_instance.train(
                        input_image,
                        batch_axes=None,
                        channel_axes=None,
                    )

                    # Denoise the masked input (J-invariance protocol)
                    denoised = denoiser_instance.denoise(
                        masked_input,
                        batch_axes=None,
                        channel_axes=None,
                    )

                    if save_denoised_images:
                        output_path, _index_counter = get_output_image_path(
                            filename,
                            operation_type="denoised",
                        )
                        base = output_path[: output_path.rfind('.')]
                        ext = output_path[output_path.rfind('.') :]
                        output_path = f"{base}_b_{denoiser_name}{ext}"
                        imwrite(denoised, output_path)

                    # Compute all metrics before appending, so a partial
                    # failure doesn't leave the lists with mismatched lengths.
                    ss_loss = float(loss_function(input_image[mask], denoised[mask]))
                    freq, crop = resolution_estimate(denoised)
                    snr = _snr_from_frequency(freq, crop)

                    ss_losses.append(ss_loss)
                    res_estimates.append(freq)
                    snrs.append(snr)

                except Exception:
                    aprint(
                        f"Denoiser '{denoiser_name}' failed on "
                        f"'{filename}' (run {run_idx + 1}/{nbruns}), skipping."
                    )
                    continue

            if ss_losses:
                self_supervised_loss_results[filename][denoiser_name] = numpy.average(
                    ss_losses
                )
                estimated_snr_results[filename][denoiser_name] = numpy.average(snrs)
                estimated_res_results[filename][denoiser_name] = numpy.average(
                    res_estimates
                )
            else:
                aprint(
                    f"All {nbruns} run(s) of '{denoiser_name}' failed "
                    f"on '{filename}'."
                )

    result_pairs = [
        ("self_supervised_loss.csv", self_supervised_loss_results),
        ("estimated_snr.csv", estimated_snr_results),
        ("res_estimate.csv", estimated_res_results),
    ]

    # Collect all denoiser names across all images for consistent CSV columns
    all_denoisers = sorted(
        set().union(*(d.keys() for d in self_supervised_loss_results.values()))
    )

    if not all_denoisers:
        aprint("No denoisers completed successfully; no CSV files written.")
        return

    # Write the results into csv files
    for output_file, result_dict in result_pairs:
        with open(output_file, 'w', newline='') as file:
            w = csv.DictWriter(file, ["filename"] + all_denoisers)
            w.writeheader()

            for key, elem in result_dict.items():
                row = {"filename": key}
                for d in all_denoisers:
                    row[d] = elem.get(d, '')
                w.writerow(row)


def handle_files(files, slicing):
    """Read and preprocess a collection of image files.

    Expands glob patterns, reads each image file using :func:`aydin.io.io.imread`,
    and applies optional NumPy-style slicing via
    :func:`aydin.util.misc.slicing_helper.apply_slicing`.

    Parameters
    ----------
    files : tuple of str
        Input file paths or glob patterns.  Shell-style wildcards (``*``,
        ``?``) are expanded internally so the function works even when the
        calling shell does not perform globbing.
    slicing : str
        NumPy-style slicing specification string (e.g., ``'0:10'``,
        ``'::2'``) applied to each loaded image array.  An empty string
        means no slicing is performed.

    Returns
    -------
    filepaths : list of str
        Expanded list of file paths (one entry per matched file).
    image_arrays : list of numpy.ndarray
        List of loaded (and optionally sliced) image arrays.
    metadatas : list of aydin.io.io.FileMetadata
        List of metadata objects for each image, containing information such
        as shape, dtype, batch axes, and channel axes.

    Raises
    ------
    click.BadParameter
        If a literal (non-glob) file path does not match any existing file.
    """
    from aydin.io.io import imread
    from aydin.util.misc.slicing_helper import apply_slicing

    filepaths = _expand_file_globs(files)
    image_arrays = []
    metadatas = []

    for filename in filepaths:
        # Get abspath to image and read it
        path = os.path.abspath(filename)

        input_image, input_metadata = imread(path)
        input_image = apply_slicing(input_image, slicing)
        image_arrays.append(input_image)
        metadatas.append(input_metadata)

    return filepaths, image_arrays, metadatas


@cli.command()
def cite():
    """Print the Aydin citation information.

    Displays the Zenodo DOI link that should be used when citing Aydin
    in academic publications.
    """
    click.echo(format_cite_box())


if __name__ == '__main__':
    cli()
