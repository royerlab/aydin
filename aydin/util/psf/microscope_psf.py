"""Gibson-Lanni microscope PSF model.

Implements a fast PSF solver using Fourier-Bessel series approximation
of the Gibson & Lanni model for fluorescence microscopy.
"""

import cmath
import math

import numpy
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.special


class MicroscopePSF:
    """Generate a PSF using the Gibson and Lanni model.

    All distance units are microns.

    Implements a fast PSF solver using Fourier-Bessel series approximation
    of the Gibson & Lanni optical path difference model. Based on the
    Python code provided by Kyle Douglass [1]_.

    Attributes
    ----------
    num_basis : int
        Number of rescaled Bessel functions used to approximate the phase.
    rho_samples : int
        Number of pupil samples along the radial direction.
    parameters : dict
        Dictionary of microscope optical parameters (magnification, NA,
        refractive indices, thicknesses, tube length).

    References
    ----------
    .. [1] K. Douglass, "Implementing a fast Gibson-Lanni PSF solver in Python".
       http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python.html
    .. [2] Li et al, "Fast and accurate three-dimensional point spread function
       computation for fluorescence microscopy", JOSA, 2017.
    .. [3] Gibson, S. & Lanni, F., "Experimental test of an analytical model of
       aberration in an oil-immersion objective lens used in three-dimensional
       light microscopy", J. Opt. Soc. Am. A 9, 154-166 (1992).
    .. [4] Kirshner et al, "3-D PSF fitting for fluorescence microscopy:
       implementation and localization application", Journal of Microscopy, 2012.
    """

    def __init__(self):
        """Initialize with default microscope parameters for a 100x/1.4NA oil objective."""

        # Internal constants.
        self.num_basis = (
            100  # Number of rescaled Bessels that approximate the phase function.
        )
        self.rho_samples = 1000  # Number of pupil sample along the radial direction.

        # Microscope parameters.
        # IMPORTANT: DO NOT CHANGE THESE DEFAULTS, IMPORTANT FOR TESTS
        self.parameters = {
            "M": 100.0,  # magnification
            "NA": 1.4,  # numerical aperture
            "ng0": 1.515,  # coverslip RI design value
            "ng": 1.515,  # coverslip RI experimental value
            "ni0": 1.515,  # immersion medium RI design value
            "ni": 1.515,  # immersion medium RI experimental value
            "ns": 1.33,  # specimen refractive index (RI)
            "ti0": 150,  # microns, working distance (immersion medium thickness) design value
            "tg": 170,  # microns, coverslip thickness experimental value
            "tg0": 170,  # microns, coverslip thickness design value
            "zd0": 200.0 * 1.0e3,
        }  # microscope tube length (in microns).

    def _calcRv(self, dxy, xy_size, sampling=2):
        """Calculate the radial distance vector (oversampled).

        Parameters
        ----------
        dxy : float
            Pixel size in microns.
        xy_size : int
            Number of pixels along x/y.
        sampling : int
            Oversampling factor for the radial vector.

        Returns
        -------
        numpy.ndarray
            Radial distance values in microns.
        """
        rv_max = math.sqrt(0.5 * xy_size * xy_size) + 1
        return numpy.arange(0, rv_max * dxy, dxy / sampling)

    def _configure(self, wvl):
        """Configure scaling factors and maximum rho for a given wavelength.

        Parameters
        ----------
        wvl : float
            Light wavelength in microns.

        Returns
        -------
        list
            [scaling_factor, max_rho] where scaling_factor is a 1D array
            and max_rho is the maximum normalized pupil radius.
        """
        mp = self.parameters

        # Scaling factors for the Fourier-Bessel series expansion
        min_wavelength = 0.436  # microns
        scaling_factor = (
            mp["NA"]
            * (3 * numpy.arange(1, self.num_basis + 1) - 2)
            * min_wavelength
            / wvl
        )

        # Not sure this is completely correct for the case where the axial
        # location of the flourophore is 0.0.
        #
        max_rho = (
            min([mp["NA"], mp["ng0"], mp["ng"], mp["ni0"], mp["ni"], mp["ns"]])
            / mp["NA"]
        )

        return [scaling_factor, max_rho]

    def deltaFocus(self, zd):
        """Return focal offset needed to compensate for camera position.

        Parameters
        ----------
        zd : float
            Actual camera position in microns.

        Returns
        -------
        float
            Focal offset in microns.
        """
        mp = self.parameters

        a = mp["NA"] * mp["zd0"] / mp["M"]  # Aperture radius at the back focal plane.
        return a * a * (mp["zd0"] - zd) / (2.0 * mp["zd0"] * zd)

    def gLXYZCameraScan(
        self, dxy, xy_size, zd, normalize=True, pz=0.0, wvl=0.6, zv=0.0
    ):
        """Calculate 3D G-L PSF by scanning the camera position.

        .. warning:: This method does not work correctly.

        Parameters
        ----------
        dxy : float
            Step size in the XY plane in microns.
        xy_size : int
            Number of pixels in X/Y.
        zd : numpy.ndarray
            Camera positions in microns.
        normalize : bool
            If True, normalize PSF to unit height.
        pz : float
            Particle z position above the coverslip in microns.
        wvl : float
            Light wavelength in microns.
        zv : float
            Relative z offset of the coverslip in microns.

        Returns
        -------
        numpy.ndarray
            3D PSF array with shape ``(zd.size, xy_size, xy_size)``.
        """
        # Calculate rv vector, this is 2x up-sampled.
        rv = self._calcRv(dxy, xy_size)

        # Calculate radial/Z PSF.
        PSF_rz = self.gLZRCameraScan(rv, zd, normalize=normalize, pz=pz, wvl=wvl, zv=zv)

        # Create XYZ PSF by interpolation.
        return self.psfRZToPSFXYZ(dxy, xy_size, rv, PSF_rz)

    def gLXYZFocalScan(
        self, dxy, xy_size, zv, normalize=True, pz=0.0, wvl=0.6, zd=None
    ):
        """Calculate 3D G-L PSF by scanning the microscope focus.

        Parameters
        ----------
        dxy : float
            Step size in the XY plane in microns.
        xy_size : int
            Number of pixels in X/Y.
        zv : numpy.ndarray
            Relative z offset values of the coverslip in microns
            (negative is closer to the objective).
        normalize : bool
            If True, normalize PSF to unit height.
        pz : float
            Particle z position above the coverslip in microns.
        wvl : float
            Light wavelength in microns.
        zd : float or None
            Actual camera position in microns. If None, the microscope
            tube length is used.

        Returns
        -------
        numpy.ndarray
            3D PSF array with shape ``(zv.size, xy_size, xy_size)``.
        """
        # Calculate rv vector, this is 2x up-sampled.
        rv = self._calcRv(dxy, xy_size)

        # Calculate radial/Z PSF.
        PSF_rz = self.gLZRFocalScan(rv, zv, normalize=normalize, pz=pz, wvl=wvl, zd=zd)

        # Create XYZ PSF by interpolation.
        return self.psfRZToPSFXYZ(dxy, xy_size, rv, PSF_rz)

    def gLXYZParticleScan(
        self, dxy, xy_size, pz, normalize=True, wvl=0.6, zd=None, zv=0.0
    ):
        """Calculate 3D G-L PSF by scanning a particle through the focus.

        Parameters
        ----------
        dxy : float
            Step size in the XY plane in microns.
        xy_size : int
            Number of pixels in X/Y.
        pz : numpy.ndarray
            Particle z positions above the coverslip in microns
            (positive values only).
        normalize : bool
            If True, normalize PSF to unit height.
        wvl : float
            Light wavelength in microns.
        zd : float or None
            Actual camera position in microns. If None, the microscope
            tube length is used.
        zv : float
            Relative z offset of the coverslip in microns.

        Returns
        -------
        numpy.ndarray
            3D PSF array with shape ``(pz.size, xy_size, xy_size)``.
        """
        # Calculate rv vector, this is 2x up-sampled.
        rv = self._calcRv(dxy, xy_size)

        # Calculate radial/Z PSF.
        PSF_rz = self.gLZRParticleScan(
            rv, pz, normalize=normalize, wvl=wvl, zd=zd, zv=zv
        )

        # Create XYZ PSF by interpolation.
        return self.psfRZToPSFXYZ(dxy, xy_size, rv, PSF_rz)

    def gLZRScan(self, pz, rv, zd, zv, normalize=True, wvl=0.6):
        """Calculate radial G-L PSF at specified radius and z positions.

        Internal function for computing the radial/axial PSF. Only one of
        ``pz``, ``zd``, or ``zv`` should be a multi-element array (the
        scanning variable).

        Parameters
        ----------
        pz : numpy.ndarray
            Particle z positions above the coverslip in microns.
        rv : numpy.ndarray
            Radial distance values in microns.
        zd : numpy.ndarray
            Camera positions in microns.
        zv : numpy.ndarray
            Relative z offset values of the coverslip in microns.
        normalize : bool
            If True, normalize PSF to unit height.
        wvl : float
            Light wavelength in microns.

        Returns
        -------
        numpy.ndarray
            2D PSF array with shape ``(n_z, n_r)`` where z is the
            scanning variable.
        """

        mp = self.parameters

        [scaling_factor, max_rho] = self._configure(wvl)
        rho = numpy.linspace(0.0, max_rho, self.rho_samples)

        a = (
            mp["NA"] * mp["zd0"] / math.sqrt(mp["M"] * mp["M"] + mp["NA"] * mp["NA"])
        )  # Aperture radius at the back focal plane.
        k = 2.0 * numpy.pi / wvl

        ti = zv.reshape(-1, 1) + mp["ti0"]
        pz = pz.reshape(-1, 1)
        zd = zd.reshape(-1, 1)

        opdt = self.OPD(rho, ti, pz, wvl, zd)

        # Sample the phase
        # phase = numpy.cos(opdt) + 1j * numpy.sin(opdt)
        phase = numpy.exp(1j * opdt)

        # Define the basis of Bessel functions
        # Shape is (number of basis functions by number of rho samples)
        J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)

        # Compute the approximation to the sampled pupil phase by finding the least squares
        # solution to the complex coefficients of the Fourier-Bessel expansion.
        # Shape of C is (number of basis functions by number of z samples).
        # Note the matrix transposes to get the dimensions correct.
        C, residuals, _, _ = numpy.linalg.lstsq(J.T, phase.T, rcond=None)

        rv = rv * mp["M"]
        b = k * a * rv.reshape(-1, 1) / zd

        # Convenience functions for J0 and J1 Bessel functions
        J0 = lambda x: scipy.special.jv(0, x)  # noqa: E731
        J1 = lambda x: scipy.special.jv(1, x)  # noqa: E731

        # See equation 5 in Li, Xue, and Blu
        denom = scaling_factor * scaling_factor - b * b
        R = (
            scaling_factor * J1(scaling_factor * max_rho) * J0(b * max_rho) * max_rho
            - b * J0(scaling_factor * max_rho) * J1(b * max_rho) * max_rho
        )
        R /= denom

        # The transpose places the axial direction along the first dimension of the array, i.e. rows
        # This is only for convenience.
        PSF_rz = (numpy.abs(R.dot(C)) ** 2).T

        # Normalize to the maximum value
        if normalize:
            PSF_rz /= numpy.max(PSF_rz)

        return PSF_rz

    def gLZRCameraScan(self, rv, zd, normalize=True, pz=0.0, wvl=0.6, zv=0.0):
        """Calculate radial G-L PSF by scanning the camera position.

        .. warning:: This method does not work correctly.

        Parameters
        ----------
        rv : numpy.ndarray
            Radial distance values in microns.
        zd : numpy.ndarray
            Camera positions in microns.
        normalize : bool
            If True, normalize PSF to unit height.
        pz : float
            Particle z position above the coverslip in microns.
        wvl : float
            Light wavelength in microns.
        zv : float
            Relative z offset of the coverslip in microns.

        Returns
        -------
        numpy.ndarray
            2D PSF array with shape ``(zd.size, rv.size)``.
        """
        pz = numpy.array([pz])
        zv = numpy.array([zv])

        return self.gLZRScan(pz, rv, zd, zv, normalize=normalize, wvl=wvl)

    def gLZRFocalScan(self, rv, zv, normalize=True, pz=0.0, wvl=0.6, zd=None):
        """Calculate radial G-L PSF by scanning the microscope focus.

        Parameters
        ----------
        rv : numpy.ndarray
            Radial distance values in microns.
        zv : numpy.ndarray
            Relative z offset values of the coverslip in microns
            (negative is closer to the objective).
        normalize : bool
            If True, normalize PSF to unit height.
        pz : float
            Particle z position above the coverslip in microns.
        wvl : float
            Light wavelength in microns.
        zd : float or None
            Actual camera position in microns. If None, the microscope
            tube length is used.

        Returns
        -------
        numpy.ndarray
            2D PSF array with shape ``(zv.size, rv.size)``.
        """

        mp = self.parameters

        if zd is None:
            zd = mp["zd0"]

        pz = numpy.array([pz])
        zd = numpy.array([zd])

        return self.gLZRScan(pz, rv, zd, zv, normalize=normalize, wvl=wvl)

    def gLZRParticleScan(self, rv, pz, normalize=True, wvl=0.6, zd=None, zv=0.0):
        """Calculate radial G-L PSF by scanning the particle position.

        Parameters
        ----------
        rv : numpy.ndarray
            Radial distance values in microns.
        pz : numpy.ndarray
            Particle z positions above the coverslip in microns
            (positive values only).
        normalize : bool
            If True, normalize PSF to unit height.
        wvl : float
            Light wavelength in microns.
        zd : float or None
            Actual camera position in microns. If None, the microscope
            tube length is used.
        zv : float
            Relative z offset of the coverslip in microns.

        Returns
        -------
        numpy.ndarray
            2D PSF array with shape ``(pz.size, rv.size)``.
        """

        mp = self.parameters

        if zd is None:
            zd = mp["zd0"]

        zd = numpy.array([zd])
        zv = numpy.array([zv])

        return self.gLZRScan(pz, rv, zd, zv, normalize=normalize, wvl=wvl)

    def OPD(self, rho, ti, pz, wvl, zd):
        """Calculate optical path difference (phase aberration) term.

        Computes the total OPD as the sum of contributions from the
        sample, immersion medium, coverslip, and camera position.

        Parameters
        ----------
        rho : numpy.ndarray
            Normalized pupil radius values.
        ti : numpy.ndarray
            Immersion medium thickness in microns.
        pz : numpy.ndarray
            Particle z position above the coverslip in microns.
        wvl : float
            Light wavelength in microns.
        zd : numpy.ndarray
            Actual camera position in microns.

        Returns
        -------
        numpy.ndarray
            Phase aberration values (k * OPD).
        """

        mp = self.parameters

        NA = mp["NA"]
        ns = mp["ns"]
        ng0 = mp["ng0"]
        ng = mp["ng"]
        ni0 = mp["ni0"]
        ni = mp["ni"]
        ti0 = mp["ti0"]
        tg = mp["tg"]
        tg0 = mp["tg0"]
        zd0 = mp["zd0"]

        a = NA * zd0 / mp["M"]  # Aperture radius at the back focal plane.
        k = 2.0 * numpy.pi / wvl  # Wave number of emitted light.

        OPDs = pz * numpy.sqrt(ns * ns - NA * NA * rho * rho)  # OPD in the sample.
        OPDi = ti * numpy.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * numpy.sqrt(
            ni0 * ni0 - NA * NA * rho * rho
        )  # OPD in the immersion medium.
        OPDg = tg * numpy.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * numpy.sqrt(
            ng0 * ng0 - NA * NA * rho * rho
        )  # OPD in the coverslip.
        OPDt = (
            a * a * (zd0 - zd) * rho * rho / (2.0 * zd0 * zd)
        )  # OPD in camera position.

        return k * (OPDs + OPDi + OPDg + OPDt)

    def psfRZToPSFXYZ(self, dxy, xy_size, rv, PSF_rz):
        """Convert a radial/axial PSF to a full 3D XYZ PSF by interpolation.

        Parameters
        ----------
        dxy : float
            Pixel size in microns.
        xy_size : int
            Number of pixels along x/y.
        rv : numpy.ndarray
            Radial distance values in microns.
        PSF_rz : numpy.ndarray
            2D radial/axial PSF with shape ``(n_z, n_r)``.

        Returns
        -------
        numpy.ndarray
            3D PSF array with shape ``(n_z, xy_size, xy_size)``.
        """
        # Create XY grid of radius values.
        c_xy = float(xy_size) * 0.5
        xy = numpy.mgrid[0:xy_size, 0:xy_size] + 0.5
        r_pixel = dxy * numpy.sqrt(
            (xy[1] - c_xy) * (xy[1] - c_xy) + (xy[0] - c_xy) * (xy[0] - c_xy)
        )

        # Create XYZ PSF by interpolation.
        PSF_xyz = numpy.zeros((PSF_rz.shape[0], xy_size, xy_size))
        for i in range(PSF_rz.shape[0]):
            psf_rz_interp = scipy.interpolate.interp1d(rv, PSF_rz[i, :])
            PSF_xyz[i, :, :] = psf_rz_interp(r_pixel.ravel()).reshape(xy_size, xy_size)

        return PSF_xyz

    def slowGL(self, max_rho, rv, zv, pz, wvl, zd):
        """Calculate a single G-L PSF point using numerical integration.

        Primarily for testing and reference. Very slow compared to the
        Fourier-Bessel approximation.

        Parameters
        ----------
        max_rho : float
            Maximum normalized pupil radius.
        rv : float
            Radial distance in microns.
        zv : float
            Relative z offset of the coverslip in microns.
        pz : float
            Particle z position above the coverslip in microns.
        wvl : float
            Light wavelength in microns.
        zd : float
            Camera position in microns.

        Returns
        -------
        float
            PSF intensity value at the specified point.
        """

        mp = self.parameters

        a = (
            mp["NA"] * mp["zd0"] / math.sqrt(mp["M"] * mp["M"] + mp["NA"] * mp["NA"])
        )  # Aperture radius at the back focal plane.
        k = 2.0 * numpy.pi / wvl
        ti = zv + mp["ti0"]

        rv = rv * mp["M"]

        def integral_fn_imag(rho):
            """Return the imaginary part of the PSF integrand at pupil radius ``rho``.

            Parameters
            ----------
            rho : float
                Normalized pupil radius.

            Returns
            -------
            float
                Imaginary part of the integrand.
            """
            t1 = k * a * rho * rv / zd
            t2 = scipy.special.jv(0, t1)
            t3 = t2 * cmath.exp(1j * self.OPD(rho, ti, pz, wvl, zd)) * rho
            return t3.imag

        def integral_fn_real(rho):
            """Return the real part of the PSF integrand at pupil radius ``rho``.

            Parameters
            ----------
            rho : float
                Normalized pupil radius.

            Returns
            -------
            float
                Real part of the integrand.
            """
            t1 = k * a * rho * rv / zd
            t2 = scipy.special.jv(0, t1)
            t3 = t2 * cmath.exp(1j * self.OPD(rho, ti, pz, wvl, zd)) * rho
            return t3.real

        int_i = scipy.integrate.quad(lambda x: integral_fn_imag(x), 0.0, max_rho)[0]
        int_r = scipy.integrate.quad(lambda x: integral_fn_real(x), 0.0, max_rho)[0]

        t1 = k * a * a / (zd * zd)
        return t1 * (int_r * int_r + int_i * int_i)

    def gLZRFocalScanSlow(self, rv, zv, normalize=True, pz=0.0, wvl=0.6, zd=None):
        """Calculate radial G-L PSF by focal scan using numerical integration.

        Slow reference implementation of ``gLZRFocalScan``.

        Parameters
        ----------
        rv : numpy.ndarray
            Radial distance values in microns.
        zv : numpy.ndarray
            Relative z offset values of the coverslip in microns.
        normalize : bool
            If True, normalize PSF to unit height.
        pz : float
            Particle z position above the coverslip in microns.
        wvl : float
            Light wavelength in microns.
        zd : float or None
            Camera position in microns. If None, uses the tube length.

        Returns
        -------
        numpy.ndarray
            2D PSF array with shape ``(zv.size, rv.size)``.
        """

        mp = self.parameters

        if zd is None:
            zd = mp["zd0"]

        [scaling_factor, max_rho] = self._configure(wvl)

        psf_rz = numpy.zeros((zv.size, rv.size))
        for i in range(zv.size):
            for j in range(rv.size):
                psf_rz[i, j] = self.slowGL(max_rho, rv[j], zv[i], pz, wvl, zd)

        if normalize:
            psf_rz = psf_rz / numpy.max(psf_rz)

        return psf_rz

    def gLZRParticleScanSlow(self, rv, pz, normalize=True, wvl=0.6, zd=None, zv=0.0):
        """Calculate radial G-L PSF by particle scan using numerical integration.

        Slow reference implementation of ``gLZRParticleScan``.

        Parameters
        ----------
        rv : numpy.ndarray
            Radial distance values in microns.
        pz : numpy.ndarray
            Particle z positions above the coverslip in microns.
        normalize : bool
            If True, normalize PSF to unit height.
        wvl : float
            Light wavelength in microns.
        zd : float or None
            Camera position in microns. If None, uses the tube length.
        zv : float
            Relative z offset of the coverslip in microns.

        Returns
        -------
        numpy.ndarray
            2D PSF array with shape ``(pz.size, rv.size)``.
        """

        mp = self.parameters

        if zd is None:
            zd = mp["zd0"]

        [scaling_factor, max_rho] = self._configure(wvl)

        psf_rz = numpy.zeros((pz.size, rv.size))
        for i in range(pz.size):
            for j in range(rv.size):
                psf_rz[i, j] = self.slowGL(max_rho, rv[j], zv, pz[i], wvl, zd)

        if normalize:
            psf_rz = psf_rz / numpy.max(psf_rz)

        return psf_rz
