"""Sample data providers for the Aydin napari plugin.

Each function returns a list of ``(data, kwargs, layer_type)`` tuples
that napari turns into layers.  These appear in napari's
**File > Open Sample > Aydin** menu.

The Zenodo-hosted samples are downloaded on first use and cached locally
(same cache as Aydin Studio's "Examples" menu).

**Exception safety**: Every public sample function catches ``BaseException``
so that errors never propagate into PyQt's slot handler.  An unhandled
``SystemExit`` inside a slot causes ``Py_Exit`` while the event loop is
still running, which crashes the process when napari's background QThreads
(e.g. ``StatusChecker``) are destroyed while still alive.
"""

import functools

import numpy
from skimage.data import binary_blobs, camera


def _normalise(image):
    """Rescale image to [0, 1] float32."""
    image = image.astype(numpy.float32)
    lo, hi = image.min(), image.max()
    if hi - lo > 0:
        image = (image - lo) / (hi - lo)
    return image


def _sample_error(name, exc):
    """Log an error and show a napari notification, return empty layer list."""
    import traceback

    traceback.print_exc()
    try:
        from napari.utils.notifications import show_error

        show_error(f'Aydin: failed to load sample "{name}": {exc}')
    except Exception:
        pass
    return []


def _safe_sample(name):
    """Decorator that catches all exceptions from a sample data provider.

    Wraps the entire function body — including imports — so that nothing
    can propagate into PyQt's slot handler.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper():
            try:
                return fn()
            except BaseException as exc:
                return _sample_error(name, exc)

        return wrapper

    return decorator


# ------------------------------------------------------------------
# Synthetic samples (no download required)
# ------------------------------------------------------------------


@_safe_sample('Noisy Camera (2D)')
def noisy_camera_sample():
    """Noisy 2D camera image (512 x 512, Poisson + Gaussian noise)."""
    from aydin.io.datasets import add_noise

    clean = _normalise(camera())
    noisy = add_noise(clean, intensity=10, variance=0.005, seed=0)
    return [(noisy, {'name': 'Noisy Camera (2D)'}, 'image')]


@_safe_sample('Noisy Blobs (3D)')
def noisy_blobs_3d_sample():
    """Noisy 3D binary blobs (64 x 128 x 128, Poisson + Gaussian noise)."""
    from aydin.io.datasets import add_noise

    clean = _normalise(
        binary_blobs(length=128, n_dim=3, blob_size_fraction=0.07, rng=1).astype(
            numpy.float32
        )
    )
    clean = clean[:64]
    noisy = add_noise(clean, intensity=10, variance=0.01, seed=0)
    return [(noisy, {'name': 'Noisy Blobs (3D)'}, 'image')]


# ------------------------------------------------------------------
# Zenodo-hosted samples (same as Aydin Studio "Examples" menu)
# ------------------------------------------------------------------


def _zenodo_sample(enum_member, name):
    """Load a Zenodo-hosted example via the datasets cache."""
    array = enum_member.get_array()

    if array is None:
        return _sample_error(name, RuntimeError('no data returned'))

    # Some PNGs load with extra channels that confuse napari:
    #   - Grayscale + Alpha (2 ch): fountain.png, monalisa.png
    #   - RGB (3 ch): Gauss_noisy.png
    # Strip to first channel since all samples are monochrome images.
    if array.ndim == 3 and array.shape[-1] in (2, 3, 4):
        array = array[..., 0]
    return [(array, {'name': name}, 'image')]


@_safe_sample('New York (noisy)')
def noisy_newyork_sample():
    """Noisy New York image (2D)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.noisy_newyork, 'New York (noisy)')


@_safe_sample('Fountain (noisy)')
def noisy_fountain_sample():
    """Noisy fountain image (2D)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.noisy_fountain, 'Fountain (noisy)')


@_safe_sample('Mona Lisa (noisy)')
def noisy_monalisa_sample():
    """Noisy Mona Lisa image (2D)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.noisy_monalisa, 'Mona Lisa (noisy)')


@_safe_sample('Gauss (noisy)')
def noisy_gauss_sample():
    """Gaussian noise test image (2D)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.noisy_gauss, 'Gauss (noisy)')


@_safe_sample('Periodic Noise')
def periodic_noise_sample():
    """Periodic noise test image (2D)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.periodic_noise, 'Periodic Noise')


@_safe_sample('Chessboard (noisy)')
def noisy_chessboard_sample():
    """Noisy chessboard image (2D, SIDD benchmark)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.noisy_brown_chessboard, 'Chessboard (noisy)')


@_safe_sample('HCR (Royer)')
def hcr_sample():
    """HCR light-sheet microscopy (3D, Royer Lab)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.royerlab_hcr, 'HCR (Royer)')


@_safe_sample('Blastocyst (Maitre)')
def maitre_mouse_sample():
    """Mouse blastocyst 3D+t time-lapse (Maitre Lab)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.maitre_mouse, 'Blastocyst (Maitre)')


@_safe_sample('OpenCell ARHGAP21 (Leonetti)')
def leonetti_arhgap21_sample():
    """OpenCell ARHGAP21 confocal stack (Leonetti Lab)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(
        examples_single.leonetti_arhgap21, 'OpenCell ARHGAP21 (Leonetti)'
    )


@_safe_sample('OpenCell ANKRD11 (Leonetti)')
def leonetti_ankrd11_sample():
    """OpenCell ANKRD11 confocal stack (Leonetti Lab)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(
        examples_single.leonetti_ankrd11, 'OpenCell ANKRD11 (Leonetti)'
    )


@_safe_sample('Drosophila Egg Chamber (Machado)')
def machado_egg_chamber_sample():
    """Drosophila egg chamber (Machado et al.)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(
        examples_single.machado_drosophile_egg_chamber,
        'Drosophila Egg Chamber (Machado)',
    )


# ------------------------------------------------------------------
# Additional Zenodo-hosted samples (beyond Studio's Examples menu)
# ------------------------------------------------------------------


@_safe_sample('Fixed Pattern Noise (Huang)')
def fixed_pattern_noise_sample():
    """Fixed pattern noise test image (3D)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(
        examples_single.huang_fixed_pattern_noise, 'Fixed Pattern Noise (Huang)'
    )


@_safe_sample('Drosophila 3D (Keller)')
def drosophila_3d_sample():
    """Drosophila melanogaster light-sheet 3D stack (Keller Lab)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.keller_dmel, 'Drosophila 3D (Keller)')


@_safe_sample('Fly Brain 3ch (Janelia)')
def flybrain_sample():
    """Fly brain 3-channel medium-size 3D stack (Janelia)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.janelia_flybrain, 'Fly Brain 3ch (Janelia)')


@_safe_sample('Tribolium nGFP (Myers)')
def tribolium_sample():
    """Tribolium nGFP light-sheet 3D stack (Myers Lab)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.myers_tribolium, 'Tribolium nGFP (Myers)')


@_safe_sample('HeLa XYZT (Hyman)')
def hela_sample():
    """HeLa cell XYZT time-lapse (Hyman Lab)."""
    from aydin.io.datasets import examples_single

    return _zenodo_sample(examples_single.hyman_hela, 'HeLa XYZT (Hyman)')
