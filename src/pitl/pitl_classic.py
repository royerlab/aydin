from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures
from src.pitl.regression.lgbm import LightGBMRegressor


class ImageTranslator:
    """
        Portable Image Translation Learning (PITL)

        Using classic ML (feature generation + regression)

    """

    def __init__(self,
                 feature_generator=MultiscaleConvolutionalFeatures(),
                 regressor=LightGBMRegressor()
                 ):
        """

        :param feature_generator:
        :type feature_generator:
        :param regressor:
        :type regressor:
        """
        self.debug_log = True
        self.models = []

        self.feature_generator = feature_generator
        self.regressor = regressor

        self.self_supervised = None

    def _compute_features(self, image, exclude_center):
        """

        :param image:
        :type image:
        :param exclude_center:
        :type exclude_center:
        :return:
        :rtype:
        """
        if self.debug_log:
            print("[RCF] computing features ")

        self.feature_generator.exclude_center = exclude_center
        features = self.feature_generator.compute(image)
        x = features.reshape(-1, features.shape[-1])

        return x

    def _predict_from_features(self,
                               x,
                               input_image_shape,
                               clip=(0, 1)):
        """
            internal function that predicts y from the features x
        :param x:
        :type x:
        :param input_image_shape:
        :type input_image_shape:
        :param clip:
        :type clip:
        :return:
        :rtype:
        """
        yp = self.regressor.predict(x)
        inferred_image = yp.reshape(input_image_shape).clip(*clip)
        return inferred_image

    def train(self,
              input_image,
              target_image,
              clip=(0, 1)):
        """
            Train to translate a given input image to a given output image
        :param input_image:
        :type input_image:
        :param target_image:
        :type target_image:
        :param clip:
        :type clip:
        :return:
        :rtype:
        """
        if self.debug_log:
            print("[RCF] training on images of dimension %s " % str(input_image.shape))

        self.self_supervised = input_image is target_image

        x = self._compute_features(input_image, self.self_supervised)

        y = target_image.reshape(-1)

        nb_features = x.shape[-1]
        nb_entries = y.shape[0]

        if self.debug_log:
            print("[RCF] Number of entries: %d features: %d" % (nb_entries, nb_features))

        # TODO: we need to fix how the test set is chosen, the first 10% voxels might not be representative (all black?)
        x_test = x[0:nb_entries // 10]
        x_train = x[nb_entries // 10:]

        y_test = y[0:nb_entries // 10]
        y_train = y[nb_entries // 10:]

        if self.debug_log:
            print("[RCF] Training...")
        self.regressor.fit(x_train, y_train, x_test=x_test, y_test=y_test)

        inferred_image = self._predict_from_features(x, input_image.shape, clip)

        if self.debug_log:
            print("[RCF] result: psnr=%f, ssim=%f " % (psnr(inferred_image, target_image), ssim(inferred_image, target_image)))

        return inferred_image

    def translate(self, input_image, clip=(0, 1)):
        """
            Translates an input image into an output image according to the learned function
        :param input_image:
        :type input_image:
        :param clip:
        :type clip:
        :return:
        :rtype:
        """
        if self.debug_log:
            print("[RCF] predicting output image from input image of dimension %s " % str(input_image.shape))

        features = self._compute_features(input_image, self.self_supervised)

        inferred_image = self._predict_from_features(features,
                                                     input_image_shape=input_image.shape,
                                                     clip=clip)
        return inferred_image
