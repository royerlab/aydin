from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures
from src.pitl.pitl_classic import ImageTranslator
from src.pitl.regression.lgbm import LightGBMRegressor


class Noise2Self:
    scales = [1, 3, 5, 11, 21, 23, 47, 95]
    widths = [3, 3, 3, 3, 3, 3, 3, 3]

    def __init__(self, scales=None, widths=None):
        if scales is not None:
            Noise2Self.scales = scales
        if widths is not None:
            Noise2Self.widths = widths

    @staticmethod
    def run(noisy_image):
        """
        Method to run Noise2Self service

        :param self:
        :param noisy_image: input noisy image, must be np compatible
        :return: denoised version of the input image, will be np compatible
        """
        generator = MultiscaleConvolutionalFeatures(kernel_widths=Noise2Self.widths[0:7],
                                                    kernel_scales=Noise2Self.scales[0:7],
                                                    kernel_shapes=['l1'] * len(Noise2Self.scales[0:7]),
                                                    exclude_center=True)

        regressor = LightGBMRegressor(learning_rate=0.01,
                                      num_leaves=256,
                                      max_depth=8,
                                      n_estimators=1024,
                                      early_stopping_rounds=20)

        it = ImageTranslator(feature_generator=generator, regressor=regressor)
        return it.train(noisy_image, noisy_image)
