from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures
from src.pitl.pitl_classic import ImageTranslator
<<<<<<< HEAD
from src.pitl.regression.lgbm import LightGBMRegressor
=======
from src.pitl.regression.gbm import GBMRegressor
>>>>>>> upstream/master


class Noise2Truth:
    scales = [1, 3, 5, 11, 21, 23, 47, 95]
    widths = [3, 3, 3, 3, 3, 3, 3, 3]

    def __init__(self, scales=None, widths=None):
        if scales is not None:
            Noise2Truth.scales = scales
        if widths is not None:
            Noise2Truth.widths = widths

    @staticmethod
    def run(noisy_image, image, noisy_test):
        # TODO: add previously trained model checks and desired behavior
        """
        Method to run Noise2Truth service

        :param noisy_image: input noisy image, must be np compatible
        :param image: input noisy image, must be np compatible
        :param noisy_test: input noisy image, must be np compatible
        :return: denoised version of the input image, will be np compatible
        """

        generator = MultiscaleConvolutionalFeatures(kernel_widths=Noise2Truth.widths,
                                                    kernel_scales=Noise2Truth.scales,
                                                    exclude_center=False)

<<<<<<< HEAD
        regressor = LightGBMRegressor(num_leaves=63,
                                      n_estimators=512)
=======
        regressor = GBMRegressor(num_leaves=63,
                                 n_estimators=512)
>>>>>>> upstream/master

        it = ImageTranslator(feature_generator=generator, regressor=regressor)

        denoised = it.train(noisy_image, image) # TODO: figure out what is going on with return
        return it.translate(noisy_test)
