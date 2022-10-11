Deconvolving an image using aydin API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One can use the following lines to deconvolve a single image with default
options using our Object-Oriented denoising API.

.. code-block:: python

   from aydin.restoration.deconvolve.lr import LucyRichardson

   lr = LucyRichardson(
        psf_kernel=psf_kernel, max_num_iterations=20, backend='scipy-cupy'
    )

   lr.train(noisy_and_blurred_image, noisy_and_blurred_image)

   lr_deconvolved_image = lr.deconvolve(noisy_and_blurred_image)



One can also use the following lines to deconvolve a single image with default
options using our procedural deconvolving endpoint.

.. code-block:: python

   from aydin.restoration.deconvolve.lr import lucyrichardson

   lr_deconvolved_image = lucyrichardson(noisy_and_blurred_image)