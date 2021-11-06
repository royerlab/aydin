

.. image:: resources/aydin_logo_grad_black.png
  :width: 400
  :alt: Logo text

*Image denoising, but chill...*

Aydin is a user-friendly, feature-rich, and fast image denoising tool that provides
a number of self-supervised, auto-tuned, and unsupervised image denoising algorithms.
Aydin handles from the get-go n-dimensional array-structured images with an arbitrary number
of batch dimensions, channel dimensions, and typically up to 4 spatio-temporal dimensions.

It comes with `Aydin Studio` a [graphical user interface]
to easily experiment with all the different algorithms and parameters available,
a [command line interface] to run large jobs offline, and an [API] for
custom coding and integration into other packages.

Getting started, with bundles:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To get started with Aydin, please download and install Aydin on your machine.


.. raw:: html

    <div style="text-align: center; margin-left: 10%; margin-right: 10%">
       <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Tux.svg/249px-Tux.svg.png" style="max-height:100px; max-width:100px; overflow: hidden; float: left">
       <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Windows_logo_-_2012_%28dark_blue%29.svg/1280px-Windows_logo_-_2012_%28dark_blue%29.svg.png" style="max-height:100px; max-width:100px; overflow: hidden; float: center">
       <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Apple_logo_black.svg/505px-Apple_logo_black.svg.png" style="max-height:100px; max-width:100px; overflow: hidden; float: right">
    </div>
    <br>
    <br>


Getting started, with pip:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install aydin


Requirements
~~~~~~~~~~~~~
While aydin works even on tiny laptops, it will run faster and better if you have
a Nvidia graphics card. We also recommend at least 16GB of RAM but more is better
especially for very large gigabyte-sized images. In the absence of a GPU,
the more CPU cores the better, obviously. Different algorithms have different performance
profiles, some of our best algorithms can easily run on modest machines.


Documentation
~~~~~~~~~~~~~~
Tutorials can be found [here] for an exhaustive tour of features and parameters for both
the graphical interface (Aydin Studio), command line interface (CLI), and
Python programming interface (API). To better understand how to tune parameters
for a given image, please check our [use cases] where we go through.

How to cite?
~~~~~~~~~~~~~
If you find Aydin useful and use it in your work, please kindly consider to cite us:

--- citation here + DOI ---



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   Install <getting_started/install.rst>
   Hardware Requirements <getting_started/hardware_requirements.rst>


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Use Cases

   Introduction <use_cases/introduction.rst>
   Denoising Basics with Aydin <use_cases/basics.rst>
   Denoising Spinning-Disk Confocal Microscopy Images with Aydin <use_cases/confocal.rst>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials

   Tutorials Home <tutorials/tutorials_home.rst>
   Aydin Studio (GUI) Tutorials <tutorials/gui_tutorials.rst>
   Aydin CLI Tutorials <tutorials/cli_tutorials.rst>
   Aydin API Tutorials <tutorials/api_tutorials.rst>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   Introduction <api/introduction>
   Restoration <api/restoration>
   Image Translator <api/image_translator>
   Transforms <api/transforms>
   Feature Generator <api/feature_generator>
   Regressors <api/regressors>
   IO <api/io>
   NN <api/nn>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contact Us

   On Github <contact_us/github>
