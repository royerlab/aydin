"""Simplified microscope PSF model without coverslip aberrations.

Provides a simplified PSF generator that models Richards & Wolf diffraction
as a degenerate case of the Gibson & Lanni model (no coverslip).
"""

import numpy

from aydin.util.psf.microscope_psf import MicroscopePSF


class SimpleMicroscopePSF(MicroscopePSF):
    """Simplified microscope PSF without coverslip aberrations.

    Models the PSF using the Richards & Wolf formulation, implemented
    as a degenerate case of the Gibson & Lanni model with matched
    refractive indices (no coverslip mismatch).

    Parameters
    ----------
    M : float
        Magnification. Default is 16.
    NA : float
        Numerical aperture. Default is 0.8.
    n : float
        Refractive index of the medium. Default is 1.33 (water).
    wd : float
        Working distance in microns. Default is 3000.
    """

    def __init__(self, M=16, NA=0.8, n=1.33, wd=3000):
        """Initialize with simplified microscope parameters."""
        super().__init__()

        # Microscope parameters.
        self.parameters["M"] = M  # magnification
        self.parameters["NA"] = NA  # numerical aperture
        self.parameters["ni0"] = n
        self.parameters["ni"] = n
        self.parameters["ns"] = n
        self.parameters["ti0"] = wd

    def generate_xyz_psf(self, dxy, dz, xy_size, z_size):
        """Generate a 3D PSF array by particle scanning.

        Parameters
        ----------
        dxy : float
            Voxel dimension along x and y in microns.
        dz : float
            Voxel dimension along z in microns.
        xy_size : int
            Size of the PSF kernel along x and y (should be odd).
        z_size : int
            Size of the PSF kernel along z (should be odd).

        Returns
        -------
        numpy.ndarray
            3D PSF array with shape (z_size, xy_size, xy_size).
        """
        lz = (z_size) * dz
        z_offset = -(lz - 2 * dz) / 2
        pz = numpy.arange(0, lz, dz)

        # gLXYZParticleScan(self, dxy, xy_size, pz, normalize = True, wvl = 0.6, zd = None, zv = 0.0):
        psf_xyz_array = self.gLXYZParticleScan(
            dxy=dxy, xy_size=xy_size, pz=pz, zv=z_offset
        )
        print(psf_xyz_array.shape)

        return psf_xyz_array
