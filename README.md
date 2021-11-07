[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![codecov](https://codecov.io/gl/aydinorg/aydin/branch/master/graph/badge.svg?token=gV3UqFAg5U)](https://codecov.io/gl/aydinorg/aydin)
# Aydin

Denoising, but chill...

Aydin is a user-friendly, feature-rich, and fast image denoising tool that provides
a number of self-supervised, auto-tuned, and unsupervised image denoising algorithms.
Aydin handles from the get-go n-dimensional array-structured images with an arbitrary number
of batch dimensions, channel dimensions, and typically up to 4 spatio-temporal dimensions.

It comes with `Aydin Studio` a [graphical user interface]
to easily experiment with all the different algorithms and parameters available,
a [command line interface] to run large jobs offline, and an [API] for
custom coding and integration into your scripts and applications.

## Supported algorithms:



<!--
This is broken right noew:
[![pipeline status](https://gitlab.com/aydinorg/aydin/badges/master/pipeline.svg)](https://gitlab.com/aydinorg/aydin/commits/master)
-->

## Documentation

Aydin documentation can be found [here](https://royerlab.github.io/aydin/).

## Installation

You can get the suitable aydin executable from [here](https://royerlab.github.io/aydin/) or you 
can do `pip install aydin`, for mac first need to do `brew install libomp`.

## How to run?

To run CLI for denoising:
```bash
aydin denoise path/to/noisyimage
```

To get help  on CLI usage:
```bash
aydin -h
```

## Recommended Environment

#### Linux

- Ubuntu 18.04+

#### Windows

- Windows 10

#### OSX

- Mojave 10.14.6+

## Contributing

Feel free to check our [contributing guideline](CONTRIBUTING.md) first and start 
discussing your new ideas and feedback with us through issues.
