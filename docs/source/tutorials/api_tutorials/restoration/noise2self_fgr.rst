Denoising an image using Noise2SelfFGR restoration API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One can use the following lines to denoise a single image with default
options using our Object-Oriented denoising API.

.. code-block:: python

   from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR

   n2s = Noise2SelfFGR()
   n2s.train(noisy_image)
   denoised_image = n2s.denoise(noisy_image)


It is also easy to pass specific transforms to use before and/or after
denoising. One can do the the following:

.. code-block:: python

   from aydin.restoration.denoise.noise2selffgr import Noise2SelfFGR

   transforms = [
        {"class": RangeTransform, "kwargs": {}},
        {"class": PaddingTransform, "kwargs": {}},
    ]
   n2s = Noise2SelfFGR(it_transforms=transforms)
   n2s.train(noisy_image)
   denoised_image = n2s.denoise(noisy_image)

One can also use the following lines to denoise a single image with default
options using our procedural denoising endpoint.

.. code-block:: python

   from aydin.restoration.denoise.noise2selffgr import noise2self_fgr

   denoised_image = noise2self_fgr(noisy_image)


