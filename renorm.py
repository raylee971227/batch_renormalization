import tensorflow as tf
from keras.layers import Layer
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
    def __init__(self, training=True, microbatch_size=1, r_max=3., d_max=5., epsilon=1e-3, momentum=0.99):
        super(BatchRenorm, self).__init__()
        self.channels = input_shape[-1]
        self.training = training
        self.microbatch_size = microbatch_size
        self.r_max = r_max
        self.d_max = d_max
        self.momentum = momentum
        self.epsilon = epsilon

        with(tf.variable_scope(None, 'batch_norm')):
            self.beta = tf.get_variable("beta", [self.channels], tf.float32, tf.zeros_initializer)
            self.gamma = tf.get_variable("gamma", [self.channels], tf.float32, tf.ones_initializer)
            self.mu = tf.get_variable("mu", [self.channels], tf.float32, trainable=False)
            self.sigma = tf.get_variable("sigma", [self.channels], tf.float32, trainable=False)

            self.mu_prev = tf.get_variable("mu_prev", [self.channels], tf.float32, initializer=tf.zeros_initializer, trainable=False)
            self.sigma_prev = tf.get_variable("sigma_prev", [self.channels], tf.float32, initializer=tf.ones_initializer, trainable=False)


        '''
            tf.cond(true or false, true function, false function)
        '''
        result = tf.cond(training, lambda: self.train(self.mu, self.sigma, momentum),
                         lambda: self.gamma * ((input - self.mu) / self.sigma) + self.beta)

        return result

    def train(self, input_shape, mu, sigma, alpha):
        # input -> input matrix
        # mu -> current moving average
        # sigma -> current moving standard deviation
        # alpha -> moving average update rate
        input_reshaped = tf.reshape(input_shape[0], self.microbatch_size, input_shape[1], input_shape[2], input_shape[3])
        batch_mean, batch_var = tf.nn.moments(input_reshaped, [0, 1, 2, 3], keep_dims=True)
        batch_stddev = tf.sqrt(batch_var)

        mu_asgn_old = tf.Variable(self.mu_prev, name="mu_save")
        mu_asgn_old.assign(mu)  # might be unnecessary

        sigma_asgn_old = tf.Variable(self.sigma_prev, name="sigma_save")
        sigma_asgn_old = tf.assign(sigma)  # might be unnecessary again. Check https://stackoverflow.com/questions/57311982/equivalent-of-tf-assign-in-tensorflow-2-0-beta1

        with(tf.control_dependencies([mu_asgn_old, sigma_asgn_old])):
            mu_add = alpha * tf.reduce_mean((batch_mean - mu_asgn_old), [0, 1, 2, 3])
            sigma_add = alpha * tf.reduce_mean((batch_stddev - sigma_asgn_old), [0, 1, 2, 3])

        # mu_asgn = tf.assign_add(mu, mu_add, name='mu_update')
        # sigma_asgn = tf.assign_add(sigma, sigma_add, name='sigma_update')

        # Update Operations
        mu_asgn = mu + mu_add
        sigma_asgn = sigma + sigma_add

        with(tf.control_dependencies([mu_asgn, sigma_asgn])):
            with(tf.variable_scope('r')):
                r = tf.stop_gradient(tf.clip_by_value(
                    batch_stddev / sigma_asgn_old,
                    1 / self.r_max, self.r_max))
            with(tf.variable_scope('d')):
                d = tf.stop_gradient(tf.clip_by_value(
                    (batch_mean - mu_asgn_old) / sigma_asgn_old,
                    -self.d_max, self.d_max))
            with(tf.variable_scope('y_train')):
                x_hat = ((input_reshaped - batch_mean) / batch_stddev) * r + d
                y = self.gamma * x_hat + self.beta

        return tf.reshape(y, tf.shape(input))

