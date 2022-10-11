===========
Why Aydin?
===========


Denoising
---------

Aydin and it is implemented to be a lightweight, robust, versatile and
highly configurable denoising package. Aydin is denoising for all.


Robust Denoising
----------------

TODO: Comparison table

Less Halucinations, More Useful Data
------------------------------------------

TODO: Figure for less halucinations

Accessible via API, CLI, GUI
----------------------------

TODO: GUI ss, CLI example command, API example snippet, all with links

Automatic blind spot detection
-------------------------------

TODO: explanation + figure

Preprocessing and Postprocessing helpers
------------------------------------------

Denoising quality can be -- in some cases -- dramatically improved by applying some carefully
crafted transformations. You can find here all these transformations and their parameters.
We recommend you test the effect of each individually, as well as the combined effect of all selected
before starting the denoising.

Deconvolution
-------------

Currently Aydin implements Lucy-Richardson Deconvolution method only and can be run with a
custom PSF provided. This can be used via CLI and API currently, not supported on GUI.

Viewing Images
--------------

Napari, nD image browser, is internally integrated to Aydin, hence, at any given time a user
can view their images in a napari window while using Aydin.

Hyperstacking
-------------

It is pretty easy to hyperstack images before denoising with Aydin.
