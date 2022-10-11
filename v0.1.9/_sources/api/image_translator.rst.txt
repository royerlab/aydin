Image Translator
-----------------

We made abstraction of image translator notion which offers a base implementation that can be extended
to perform any image translation task. Currently, we are offering multiple denoising and deconvolving
implementations of this base image translator class.Aydin provides a relatively generic image translator
API. By providing the required complementary parts one can use Aydin image translator module to implement
most of the image translation methods out there. Below you can find a summary of image translator
implementations in aydin currently:

.. toctree::
   :maxdepth: 1

   ImageTranslatorBase <image_translators/base>
   ImageDenoiserClassic <image_translators/classic>
   ImageTranslatorFGR <image_translators/fgr>
   ImageTranslatorCNN <image_translators/cnn>


