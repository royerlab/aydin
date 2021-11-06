#!/usr/bin/env python
"""
Test focal scan G-L PSF.
"""
from pprint import pprint

from aydin.util.psf.simple_microscope_psf import SimpleMicroscopePSF


def demo_xyz():
    """
    Particle scan, focus offset.
    """
    psf = SimpleMicroscopePSF()
    pprint(psf.parameters)

    psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)

    print(psf_xyz_array.shape)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(psf_xyz_array, name='fast_rz')


demo_xyz()
