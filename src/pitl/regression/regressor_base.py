from abc import ABC, abstractmethod


class RegressorBase(ABC):
    """
        Image Translator base class

    """

    def __init__(self):
        """

        """

    @abstractmethod
    def reset(self):
        """
        resets the regressor to a blank state.

        :param x_train: x training values
        :type x_train:
        :param y_train: y training values
        :type y_train:
        :param x_test:
        :type x_test: x test values
        :param y_test:
        :type y_test: y test values
        """
        raise NotImplementedError()


    def _batch_split_fit(self, x_train, y_train, x_valid, y_valid):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the validation dataset: (x_valid, y_valid).

        This method can be called multiple times with different batches.
        To reset the regressor call reset()

        :param x_train: x training values
        :type x_train:
        :param y_train: y training values
        :type y_train:
        :param x_valid:  x validation values
        :type x_valid:
        :param y_valid:  y validation values
        :type y_valid:
        """
        #TODO: we split the train data into batches automatically here

    @abstractmethod
    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).


        :param x_train: x training values
        :type x_train:
        :param y_train: y training values
        :type y_train:
        :param x_valid:  x validation values
        :type x_valid:
        :param y_valid:  y validation values
        :type y_valid:
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x):
        """
        Predicts y given x by applying the learned function f: y=f(x)

        :param x: x values
        :type x:
        :return: inferred y values
        :rtype:
        """
        raise NotImplementedError()

