class ImageTranslator:
    """
        Portable Image Translation Learning (PITL)

        Using CNN (Unet and Co)

    """

    def __init__(self):
        """

        :param feature_generator:
        :type feature_generator:
        :param regressor:
        :type regressor:
        """
        self.debug_log = True


        self.self_supervised = None


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

        # TODO: do something...

        return None

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

        # TODO: do something...

        return None
