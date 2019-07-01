import tempfile

import numpy
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from pitl.it.it_base import ImageTranslatorBase
from pitl.regression.gbm import GBMRegressor
from src.pitl.features.mcfocl import MultiscaleConvolutionalFeatures


class ImageTranslatorClassic(ImageTranslatorBase):
    """
        Portable Image Translation Learning (PITL)

        Using classic ML (feature generation + regression)

    """

    def __init__(self,
                 feature_generator=MultiscaleConvolutionalFeatures(),
                 regressor=GBMRegressor()
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

    def _compute_features(self, image, exclude_center, batch_dims):
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
        features = self.feature_generator.compute(image, batch_dims)
        x = features.reshape(-1, features.shape[-1])

        return x

    def _predict_from_features(self,
                               x,
                               input_image_shape):
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
        inferred_image = yp.reshape(input_image_shape)
        return inferred_image

    def train(self,
              input_image,
              target_image,
              batch_dims       = None,
              train_test_ratio = 0.1):
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
            print("[RCF] training on images of dimension %s ." % str(input_image.shape))

        self.self_supervised = input_image is target_image

        x = self._compute_features(input_image, self.self_supervised, batch_dims)

        y = target_image.reshape(-1)

        nb_features = x.shape[-1]
        nb_entries = y.shape[0]

        if self.debug_log:
            print("[RCF] Number of entries: %d features: %d ." % (nb_entries, nb_features))

        if self.debug_log:
            print("[RCF] splitting train and test sets.")
        # creates random complementary indices for selecting train and test entries:
        test_size = int(train_test_ratio*nb_entries)
        train_indices = numpy.full(nb_entries, False)
        train_indices[test_size:] = True
        numpy.random.shuffle(train_indices)
        test_indices= numpy.logical_not(train_indices)

        # we allocate memory for the new arrays taking into account that we might need to use memory mapped files
        # in the splitting of train and test sets. The features are the heavy part, so that's what we map:
        x_train, y_train, x_test, y_test = (None,)*4
        if isinstance(x, numpy.memmap):
            temp_file = tempfile.TemporaryFile()
            x_train = numpy.memmap(temp_file,
                              dtype=numpy.float32,
                              mode='w+',
                              shape=((nb_entries-test_size),nb_features))
        else:
            x_train = numpy.zeros(((nb_entries-test_size),nb_features), dtype=numpy.float)

        y_train = numpy.zeros((nb_entries-test_size,), dtype=numpy.float)
        x_test = numpy.zeros((test_size,nb_features), dtype=numpy.float)
        y_test = numpy.zeros((test_size,), dtype=numpy.float)


        # train data
        numpy.copyto(x_train, x[train_indices])
        numpy.copyto(y_train, y[train_indices])

        # test data:
        numpy.copyto(x_test, x[test_indices])
        numpy.copyto(y_test, y[test_indices])

        if self.debug_log:
            print("[RCF] Training...")
        self.regressor.fit(x_train, y_train, x_valid=x_test, y_valid=y_test)

        inferred_image = self._predict_from_features(x, input_image.shape)

        if self.debug_log:
            print("[RCF] result: psnr=%f, ssim=%f " % (psnr(inferred_image, target_image), ssim(inferred_image, target_image)))

        return inferred_image

    def translate(self, input_image, batch_dims=None):
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
            print("[RCF] predicting output image from input image of dimension %s ." % str(input_image.shape))

        features = self._compute_features(input_image, self.self_supervised, batch_dims)

        inferred_image = self._predict_from_features(features,
                                                     input_image_shape=input_image.shape)
        return inferred_image
