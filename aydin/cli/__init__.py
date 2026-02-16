"""Command-line interface for Aydin image denoising.

This package provides the Click-based CLI entry point for Aydin, exposing
subcommands for denoising, viewing, analysis (SSIM, PSNR, MSE, FSC),
channel splitting, hyperstacking, and benchmarking of denoising algorithms.

The main CLI group is defined in :mod:`aydin.cli.cli` and registered as the
``aydin`` console script in ``pyproject.toml``.
"""
