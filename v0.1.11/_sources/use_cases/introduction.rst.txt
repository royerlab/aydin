Aydin Use-Cases, and the Art & Science of Image Denoising.
===========================================================

While designing Aydin, we came to the realisation that there is no silver bullet, there is not a single algorithm nor
single set of parameters that can reliably denoise all images. More fundamentally, ‘image denoising’ is not a
singular problem, but a multiplicity of challenges. There are many different kinds of image noises and related
degradations and it requires a lot of know-how to both understand this diversity and know what to do for each kind of
noise. This diversity comes from the inherent diversity of imaging modalities: different instruments, optics,
detectors, contrast mechanisms, photo-chemistry, etc… This extends well beyond just optical images into other imaging
modalities -- it essentially holds true for any mechanism that generates images and more generally any
multi-dimensional array-like measurement.

The use-cases presented below are the ideal starting point to understand how to choose among denoising algorithms, among pre- and
post- processing steps, and how to adjust their parameters. We are actively populating this list with more datasets,
and further improving this material, please check this page for updates!

#. `Denoising Basics with Aydin <basics.html>`_
#. `Denoising Spinning-Disk Confocal Microscopy Images with Aydin <confocal.html>`_

Note: We are always interested in learning about new challenging images, please contact us by filling an issue
`here <https://github.com/royerlab/aydin/issues>`_ if you face difficulties or have a dataset that resists our methods.