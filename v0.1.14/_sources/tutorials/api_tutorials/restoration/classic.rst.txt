Denoising an image using Classic restoration API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One can use the following lines to denoise a single image with default
options using our Object-Oriented denoising API.

.. code-block:: python

   from aydin.restoration.denoise.classic import Classic

   classic_restoration = Classic()
   classic_restoration.train(noisy_image)
   denoised_image = classic_restoration.denoise(noisy_image)


It is also easy to pass specific transforms to use before and/or after
denoising. One can do the the following:

.. code-block:: python

   from aydin.restoration.denoise.classic import Classic

   transforms = [
        {"class": RangeTransform, "kwargs": {}},
        {"class": PaddingTransform, "kwargs": {}},
    ]
   classic_restoration = Classic(it_transforms=transforms)
   classic_restoration.train(noisy_image)
   denoised_image = classic_restoration.denoise(noisy_image)

One can also use the following lines to denoise a single image with default
options using our procedural denoising endpoint.

.. code-block:: python

   from aydin.restoration.denoise.classic import classic_denoise

   denoised_image = classic_denoise(noisy_image)


