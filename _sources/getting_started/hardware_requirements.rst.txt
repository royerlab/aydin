======================
Hardware Requirements
======================

Aydin tries to make use of high-end NVIDIA GPUs (Graphical Processing Units) whenever it can.
Some of the computational packages that Aydin depends on can make use of an
existing CUDA GPU out of the box (CatBoost) whereas other libraries
such as Numba require the CUDA toolkit to be installed (conda install cudatoolkit)
to be able to make use of CUDA GPUs. Aside from the CUDA GPU support, having a faster CPU with more cores
and a bigger memory can help with Aydin's runtime performance.

Recommended specifications are: at least 16 Gb of RAM, ideally 32 Gb, and more for very large
images, a CPU with at least 4 cores, and a recent NVIDIA graphics card such as a RTX series card.
Older graphics cards could work but may cause trouble or be too slow. Aydin Studio's summary page
gives an overview of the strengths and weaknesses of your machine, highlighting in red and orange
items that might be problematic.


.. image:: ../resources/system_summary_ss.png


Having said that, some algorithms in Aydin such as the 'Butterworth' denoiser  can be quite fast,
can run on modest machines, and may be sufficient for your needs.

