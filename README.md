![image](https://user-images.githubusercontent.com/1870994/140651325-711b6c30-133d-45ba-a794-8a10a4cafbc2.png?width=200)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aydin)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/royerlab/aydin/branch/main/graph/badge.svg?token=gV3UqFAg5U)](https://codecov.io/gh/royerlab/aydin)
[![DOI](https://zenodo.org/badge/188953977.svg)](https://zenodo.org/badge/latestdoi/188953977)
[![Downloads](https://pepy.tech/badge/aydin)](https://pepy.tech/project/aydin)


[graphical user interface]: https://royerlab.github.io/aydin/v0.1.15/tutorials/gui_tutorials.html
[command line interface]: https://royerlab.github.io/aydin/v0.1.15/tutorials/cli_tutorials.html
[API]: https://royerlab.github.io/aydin/v0.1.15/tutorials/api_tutorials.html
[use cases]: https://royerlab.github.io/aydin/v0.1.15/use_cases/introduction.html
[install]: https://royerlab.github.io/aydin/v0.1.15/getting_started/install.html

*Aydin* is a user-friendly, feature-rich, and fast **image denoising tool** that provides
a number of **self-supervised, auto-tuned, and unsupervised** image denoising algorithms.
*Aydin* handles from the get-go n-dimensional array-structured images with an arbitrary number
of batch dimensions, channel dimensions, and typically up to 4 spatio-temporal dimensions.

It comes with *Aydin Studio* a [graphical user interface][graphical user interface]
to easily experiment with all the different algorithms and parameters available,
a [command line interface] to run large jobs on the terminal possibly on powerful remote machines,
and an [API] for custom coding and integration into your scripts and applications.
More details and exhaustive explanations can be found in Aydin's [documentation](https://royerlab.github.io/aydin/).

## Supported algorithms:

Currently *Aydin* supports two main families of denoisers: the first family consists
of 'classical' denoising algorithms that leverage among other: frequency domain filtering,
smoothness priors, low-rank representations, self-similarity, and more.
The second family consists of algorithms that leverage machine learning approaches
such as convolutional neural networks (CNN) or gradient boosting (GB).
In the [Noise2Self paper](https://deepai.org/publication/noise2self-blind-denoising-by-self-supervision)
we show that it is possible to calibrate any parameterised denoising algorithm,
from the few parameters of a classical algorithm to the millions of weights of a deep neural
network. We leverage and extend these ideas in *Aydin* to provide a variety of auto-tuned
and trained high-quality image denoisers. What this means is that for example, we can discover automatically
the optimal parameters for non-local-means (NLM) denoising, or the best cut-off frequencies for a low-pass denoiser.
These parameters are difficult to determine 'by-hand' but when auto-tuned we show (see [use-cases][use cases])
that you can get remarkable results even with simple 'classic' denoisers, and even be competitive against more complex and slower
approaches such as deep-learning based denoisers that can also be prone to hallucination and 'copy-paste' effects.
Importantly, our experience denoising many different kinds of images has shown that there is not a single 'silver-bullet'
denoiser, different kinds of datasets require different approaches.
Here is the list of currently available methods:

- **Low-pass filtering based algorithms:**
  - Butterworth denoiser (*butterworth*).
  - Gaussian blur denoiser (*gaussian*).
  - Gaussian-Median mixed denoiser (*gm*).

- **Optimisation-based with smoothness priors:**
  - Total-Variation denoising (*tv*)
  - Harmonic prior (*harmonic*)

- **Spectral and wavelet domain:**
  - Spectral denoising (*spectral*)
  - Wavelet denoising (*wavelet*)
  - PCA denoising (*pca*)

- **Low-rank representations:**
  - Denoising via sparse decomposition (e.g. OMP) over a fixed dictionary (DCT, DST, ...)
  - Denoising via sparse decomposition (e.g. OMP) over a learned dictionary (Kmeans, PCA, ICA, SDL, ...)

- **Edge-preserving:**
  - Bilateral denoising (*bilateral*)

- **Patch similarity:**
  - Non-Local Means denoising (*nlm*)
  - BMnD -- Block-Matching nD denoising, a generalization of BM3D (*bmnd*)

- **Machine learning based:**
  - Noise2Self-FGR: Noise2Self denoising via Feature Generation and Regression (FGR). We use specially crafted integral features. We have several variants that leverage different regressors: CatBoost(cb), lightGBM, linear, perceptron, random-forest, support vector regression)
  - Noise2Self-CNN: Noise2Self denoising via Convolutional Neural Networks (CNN) using PyTorch. This is the original approach of Noise2Self. In our experience this is typically slower to train, and more prone to hallucination and residual noise than FGR.

- **Other:**
  - Lipschitz continuity denoising.

Some methods actually do combine multiple ideas and so the classification above is not strict.
We recommend trying first a good baseline denoiser such as the *Butterworth denoiser*.
If unsatisfied with the result, and you have a powerful computer with a recent NVIDIA
graphics card, then we recommend you try the Noise2Self-FGR-cb denoiser.
For detailed use-cases check [here][use cases].

We regularly come up with new approaches and ideas, there is just not enough time to write papers about all these ideas.
This means that the best 'publication' for some of these novel algorithms is this repo itself, and so please be so kind as to
[cite this repo](https://github.com/royerlab/aydin#cite-this-repo) for any ideas that you would use or reuse.
We have a long todo list of existing, modified, as well as original algorithms that we plan to add to *Aydin* in the next weeks and months. We will do so progressively as time allows. Stay tuned!

## Documentation

*Aydin*'s documentation can be found [here](https://royerlab.github.io/aydin/).

## Installation of *Aydin Studio*

We recommend that users that are not familiar with python start with our user-friendly UI.
The latest releases and standalone executables can be found on the
[releases page](https://github.com/royerlab/aydin/releases).
Detailed installation instructions of *Aydin Studio* for all three operating systems can be found
[here][install].


## Installation

### Install from PyPI

*Aydin* requires Python 3.9 or later and NumPy 2.0+. We recommend using a virtual environment:
```bash
pip install aydin
```

### Install for development

The project uses [hatchling](https://hatch.pypa.io/) as its build backend (configured in `pyproject.toml`).

```bash
git clone https://github.com/royerlab/aydin.git
cd aydin
make setup  # or: pip install -e ".[dev]"
```

Run `make help` to see all available development commands (testing, formatting, building, etc.).

### Platform-specific dependencies

**macOS:** Install OpenMP support:
```bash
brew install libomp
```
You can install *Brew* by following the instructions [here](https://brew.sh/).

**Linux:** Install the Qt system dependency (required by Qt 6.5+):
```bash
sudo apt install libxcb-cursor0    # Ubuntu/Debian
sudo dnf install xcb-util-cursor   # Fedora/RHEL
sudo pacman -S xcb-util-cursor     # Arch Linux
```

### GPU acceleration

Aydin uses PyTorch for CNN-based denoising. To enable GPU acceleration, ensure your
PyTorch installation supports CUDA. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
for platform-specific instructions.

## How to run ?

Assuming that you have installed *Aydin* in an environment, you can:

Start *Aydin Studio* from the command line with:
```bash
aydin
```

Run the Command Line Interface (CLI) for denoising:
```bash
aydin denoise path/to/noisyimage
```

Get help on command line usage:
```bash
aydin -h
```

## Recommended Operating System Versions

#### Linux: Ubuntu 20.04+
#### Windows: Windows 10+
#### macOS: Big Sur 11+

## Recommended Hardware:

Recommended specifications are: at least 16 Gb of RAM, ideally 32 Gb, and more for very large
images, a CPU with at least 4 cores, preferably 16 or more, and a recent NVIDIA graphics card such as a RTX series card.
Older graphics cards could work but may cause trouble or be too slow. *Aydin* Studio's summary page
gives an overview of the strengths and weaknesses of your machine, highlighting in red and orange
items that might be problematic.

## Known Issues:

  - On Ubuntu and perhaps other Linux systems, high-dpi modes tend to mess with font and UI element rendering.

## Contributing

Feel free to check our [contributing guideline](CONTRIBUTING.md) first and start
discussing your new ideas and feedback with us through issues.

## Cite this repo

You can cite our work with: [https://doi.org/10.5281/zenodo.5654826](https://doi.org/10.5281/zenodo.5654826)
