from abc import ABC, abstractmethod


class ImageTranslatorBase(ABC):
    """
        Image Translator base class

    """

    def __init__(self):
        """

        """
        self.debug = True
        self.self_supervised = None

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
