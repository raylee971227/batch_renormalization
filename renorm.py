import tensorflow as tf
import keras
from keras import Layer
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar100
from keras import regularizers, optimizers
from keras.utils import multi_gpu_model
import numpy as np

class BatchRenorm(Layer):
    def __init__(self, input, training, r_max, d_max, momentum=0.99, microbatch_size):
        self.channels = input.shape[-1]

