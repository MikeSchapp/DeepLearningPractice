import tensorflow as tf


class DeepLearning:
    def __init__(self, convoluted: int, fully_connected: int, epoch: int, lr: float=0.001):
        """
        For creating a Neural Network with training data, and values to predict in order to train
        :param convoluted:
        :param fully_connected:
        :param epoch:
        :param lr:
        """
        # convoluted nodes
        self.convoluted = convoluted
        # fully connected
        self.fully_connected = fully_connected
        # learning rate
        self.lr = lr
        # epochs
        self.epoch = epoch
        self.training_data = []

    def add_training_data(self):
        self.training_data.append(train)

    def del_training_data(self):
        self.training_data.remove(train)
