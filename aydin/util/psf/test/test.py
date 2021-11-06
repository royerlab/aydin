#!/usr/bin/env python
"""
Test focal scan G-L PSF.
"""
import numpy

from aydin.util.psf.microscope_psf import MicroscopePSF


def test_01():
    """
    Particle on surface.
    """
    psf = MicroscopePSF()
    rv = numpy.arange(0.0, 1.01, 0.1)
    zv = numpy.arange(-1.0, 1.01, 0.2)

    fast_rz = psf.gLZRFocalScan(rv, zv)
    slow_rz = psf.gLZRFocalScanSlow(rv, zv)

    assert numpy.allclose(fast_rz, slow_rz)


def test_02():
    """
    Particle above surface.
    """
    psf = MicroscopePSF()
    rv = numpy.arange(0.0, 1.01, 0.1)
    zv = numpy.arange(-1.0, 1.01, 0.2)

    fast_rz = psf.gLZRFocalScan(rv, zv, pz=0.5)
    slow_rz = psf.gLZRFocalScanSlow(rv, zv, pz=0.5)

    assert numpy.allclose(fast_rz, slow_rz, atol=1.0e-4, rtol=1.0e-4)


def test_03():
    """
    Detector offset.
    """
    psf = MicroscopePSF()
    rv = numpy.arange(0.0, 1.01, 0.1)
    zv = numpy.arange(-1.0, 1.01, 0.2)

    zd = psf.parameters["zd0"] + 1000
    fast_rz = psf.gLZRFocalScan(rv, zv, zd=zd)
    slow_rz = psf.gLZRFocalScanSlow(rv, zv, zd=zd)

    assert numpy.allclose(fast_rz, slow_rz)


def test_04():
    """
    Particle scan.
    """
    psf = MicroscopePSF()
    rv = numpy.arange(0.0, 1.01, 0.1)
    pv = numpy.arange(0.0, 2.01, 0.1)

    fast_rz = psf.gLZRParticleScan(rv, pv)
    slow_rz = psf.gLZRParticleScanSlow(rv, pv)

    assert numpy.allclose(fast_rz, slow_rz, rtol=1.0e-4, atol=1.0e-4)


def test_05():
    """
    Particle scan, focus offset.
    """
    psf = MicroscopePSF()
    rv = numpy.arange(0.0, 1.01, 0.1)
    pv = numpy.arange(1.0, 3.01, 0.2)

    fast_rz = psf.gLZRParticleScan(rv, pv, zv=-2.0)
    slow_rz = psf.gLZRParticleScanSlow(rv, pv, zv=-2.0)

    assert numpy.allclose(fast_rz, slow_rz, rtol=1.0e-3, atol=1.0e-3)
