![image](https://user-images.githubusercontent.com/1870994/140651325-711b6c30-133d-45ba-a794-8a10a4cafbc2.png?width=200)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![codecov](https://codecov.io/gl/aydinorg/aydin/branch/master/graph/badge.svg?token=gV3UqFAg5U)](https://codecov.io/gl/aydinorg/aydin)

Aydin is a user-friendly, feature-rich, and fast image denoising tool that provides
a number of self-supervised, auto-tuned, and unsupervised image denoising algorithms.
Aydin handles from the get-go n-dimensional array-structured images with an arbitrary number
of batch dimensions, channel dimensions, and typically up to 4 spatio-temporal dimensions.

It comes with `Aydin Studio` a [graphical user interface]
to easily experiment with all the different algorithms and parameters available,
a [command line interface] to run large jobs on the terminal possibly on powerfull remote machines, 
and an [API] for custom coding and integration into your scripts and applications.

And, of course, a simplified [napari](napari.org) plugin is in the works! 
Notebooks for running on Collab are also planned.

## Supported algorithms:

Currently Aydin support two main families of denoisers: the first family consists of 'classical' denoising algorithms that leverage among other: frequency domain filtering, smoothness priors, low-rank representations, self-similarity. The second family consists of algorithms that leverage machine learning approaches such as convolutional neural networks (CNN) or gradient boosting (GB). In the [Noise2Self paper](https://deepai.org/publication/noise2self-blind-denoising-by-self-supervision) we show that it is possible to calibrate any parameterised denoising algorithm, from the few parameters of a classical algorithm to the millions of weights of a deep neural network. We leverage this in Aydin to provide 
Here is the list of currently available algorithms: 

- Low-pass filtering based algorithms:
  - Butterworth denoiser (*butterworth*).
  - Gaussian blur denoiser (*gaussian*).
  - Gaussian-Median mixed denoiser (*gm*).
 
- Optimisation-based with smoothness priors:
  - Total-Variation denoising (*tv*)
  - Harmonic prior (*harmonic*)

- Low-rank representations: 
  - Denoising via sparse decomposition (e.g. OMP) over a fixed dictionary (DCT, DST, ...)
  - Denoising via sparse decomposition (e.g. OMP) over a learned dictionary (Kmeans, PCA, ICA, SDL, ...)

- Patch similarity:
  - Non-Local Means denoising (*nlm*)
  - BMnD (not available just yet but partly implemented!) 

- Machine learning based:
  - Noise2Self-FGR: Noise2Self denoising via Feature Generation and Regression (FGR). We use specially crafted integral features. We have several variants that leverage different regressors: CatBoost(cb), lightGBM, linear, perceptron, random-forrest, support vector regression) 
  - Noise2Self-CNN: Noise2Self denoising via Convolutional Neural Networks (CNN). This is the original approach of Noise2Self. In our experience this is typically slower to train, and more prone to hallucination and residual noise than FGR.  
 
- Other:
  - Lipschitz continuity denoising   

Some methods above actually do combine
We recomend to try first some of the classical algorithms such as 'Butterworth' or ''

We are regularly adding new algorithms so stay tuned!

<!--
This is broken right noew:
[![pipeline status](https://gitlab.com/aydinorg/aydin/badges/master/pipeline.svg)](https://gitlab.com/aydinorg/aydin/commits/master)
-->

## Documentation

Aydin documentation can be found [here](https://royerlab.github.io/aydin/).

## Installation of *Aydin Studio*

We recommend that users that are not familiar with python start with our user-friendly UI. Just download the version for your operating system:

[<img src="https://user-images.githubusercontent.com/1870994/140653991-fb570f5a-bc6f-4afd-95b6-e36d05d1382d.png" width="200" >
](https://github.com/royerlab/aydin/releases/download/v0.1.8/aydin_0.1.8_linux.zip)   
[<img src="https://user-images.githubusercontent.com/1870994/140653995-5055e607-5226-4b76-8cc4-04de17d2811f.png" width="200" >
](https://github.com/royerlab/aydin/releases/download/v0.1.8/aydin_0.1.8_win.zip)   
[<img src="https://user-images.githubusercontent.com/1870994/140653999-5f6368d9-3e82-4d10-9283-2359aa1464fa.png" width="200" >
](https://github.com/royerlab/aydin/releases/download/v0.1.8/aydin_0.1.8_osx.pkg)

## Installation of *Aydin* in Conda:

First download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://www.anaconda.com/products/individual). 

Then create a conda environment
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

## Recommended Hardware:

Recommended specifications are: at least 16 Gb of RAM, ideally 32 Gb, and more for very large
images, a CPU with at least 4 cores, preferably 16 or more, and a recent NVIDIA graphics card such as a RTX series card.
Older graphics cards could work but may cause trouble or be too slow. Aydin Studio's summary page
gives an overview of the strengths and weaknesses of your machine, highlighting in red and orange
items that might be problematic.

## Known Pressing Issues:
Here are some issues that are being actively addressed and will be addressed asap:

  - Stop button for all algorithms. For technical reasons having to do with the diversity of libraries we use, we currently cannot stop training. We are planning to solve this using subprocess spawning.

## Road Map:

Planned features:
  - Toggling between 'Advanced' and 'Basic' modes to show and hide advanced algorithms.
  - Loading of denoising model and configurations (JSON) on Aydin Studio
  - Pytorch backend

## Contributing

Feel free to check our [contributing guideline](CONTRIBUTING.md) first and start 
discussing your new ideas and feedback with us through issues.
