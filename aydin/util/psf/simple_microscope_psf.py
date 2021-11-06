import numpy
from aydin.util.psf.microscope_psf import MicroscopePSF


class SimpleMicroscopePSF(MicroscopePSF):
    """
    Simple Microscope PSF: no cover-slip, i.e. Richards & Wolf as a degenerate case of Gibson & Lani
    """

    def __init__(self, M=16, NA=0.8, n=1.33, wd=3000):
        """ """
        super().__init__()

        # Microscope parameters.
        self.parameters["M"] = M  # magnification
        self.parameters["NA"] = NA  # numerical aperture
        self.parameters["ni0"] = n
        self.parameters["ni"] = n
        self.parameters["ns"] = n
        self.parameters["ti0"] = wd

    def generate_xyz_psf(self, dxy, dz, xy_size, z_size):
        """
        Generates a 3D PSF array.

        :param dxy: voxel dimension along xy (microns)
        :param dz: voxel dimension along z (microns)
        :param xy_size: size of PSF kernel along x and y (odd integer)
        :param z_size: size of PSF kernel along z (odd integer)

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
