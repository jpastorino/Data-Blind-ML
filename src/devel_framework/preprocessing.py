from abc import ABC, abstractmethod


class Preprocessing(ABC):

    @abstractmethod
    def prepare(self, data):
        """
        Receives an np.array and preprocess it in order to be useful for a DL model.
        Available libraries are:
            - numpy as np
            - tensorflow as tf
            - sklearn as scikit

        :return: (x,y). preprocessed x (for inputs) and y (for targets)
        """
