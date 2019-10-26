from keras import backend as K
from keras import initializers, regularizers, constraints
import tensorflow as tf
from keras.engine import Layer, InputSpec
from keras.utils.generic_utils import get_custom_objects


class BatchRN(Layer):
    def __init__(self, r_max, d_max, training=True, momentum=0.99, microbatch_size=1, epsilon=0.0001):
        super(BatchRN, self).__init__()
        self.r_max = r_max
        self.d_max = d_max
        self.momentum = momentum
        self.microbatch_size = microbatch_size
        self.epsilon = epsilon
        self.training = training

    def build(self, input_shape):
        channels = input_shape[-1]
        self.beta = self.add_weight(name="beta", shape=[channels], dtype=tf.float32, initializer='zeros')
        self.gamma = self.add_weight(name="gamma", shape=[channels], dtype=tf.float32, initializer=initializers.get('ones'))
        self.mu = self.add_weight(name="mu", shape=[channels], dtype=tf.float32, initializer=initializers.get('zeros'), trainable=False)
        self.sigma = self.add_weight(name="sigma", shape=[channels], dtype=tf.float32, initializer=initializers.get('ones'), trainable=True)
        self.mu_old = self.add_weight(name="mu_old", shape=[channels], dtype=tf.float32, initializer=initializers.get('zeros'), trainable=False)
        self.sigma_old = self.add_weight(name="sigma_old", shape=[channels], dtype=tf.float32, initializer=initializers.get('ones'), trainable=False)
        # self.beta = self.add_weight(name="beta", shape=[channels], dtype=tf.float32)
        # self.gamma = self.add_weight(name="gamma", shape=[channels], dtype=tf.float32)
        # self.mu = self.add_weight(name="mu", shape=[channels], dtype=tf.float32, trainable=False)
        # self.sigma = self.add_weight(name="sigma", shape=[channels], dtype=tf.float32, trainable=True)
        # self.mu_old = self.add_weight(name="mu_old", shape=[channels], dtype=tf.float32, trainable=False)
        # self.sigma_old = self.add_weight(name="sigma_old", shape=[channels], dtype=tf.float32, trainable=False)

    def call(self, inputs, **kwargs):
        return inputs
        # if self.training in {0, False}:
        #     return self.gamma * ((inputs - self.mu) / self.sigma) + self.beta
        # else:
        #     # print("In training phase")
        #     return self.train_step(inputs)

    def train_step(self, x):
        x_shape = tf.shape(x)
        x_shaped = tf.reshape(x, [x_shape[0] // self.microbatch_size,
                                  self.microbatch_size,
                                  x.shape[1],
                                  x.shape[2],
                                  x.shape[3]])  # [N M H W C]
        mu_b, sigma_sq_b = tf.nn.moments(x_shaped, [1, 2, 3],
                                         keepdims=True)  # [N 1 1 1 C]
        sigma_b = tf.sqrt(sigma_sq_b)  # [N 1 1 1 C]

        mu_asgn_old = self.mu_old.assign(self.mu)
        sigma_asgn_old = self.sigma_old.assign(self.sigma)

        with(tf.control_dependencies([mu_asgn_old, sigma_asgn_old])):
            mu_add = self.momentum * tf.reduce_mean((mu_b - mu_asgn_old), [0, 1, 2, 3])  # [C]
            sigma_add = self.momentum * tf.reduce_mean((sigma_b - sigma_asgn_old), [0, 1, 2, 3])  # [C]
            mu_asgn = self.mu.assign_add(mu_add)
            sigma_asgn = self.mu.assign_add(sigma_add)

            with(tf.control_dependencies([mu_asgn, sigma_asgn])):
                r = tf.stop_gradient(tf.clip_by_value(
                    sigma_b / sigma_asgn_old,
                    1 / self.r_max, self.r_max))
                d = tf.stop_gradient(tf.clip_by_value(
                    (mu_b - mu_asgn_old) / sigma_asgn_old,
                    -self.d_max, self.d_max))
                x_hat = ((x_shaped - mu_b) / sigma_b) * r + d
                y = self.gamma * x_hat + self.beta

        return tf.reshape(y, x_shape)


# x = tf.ones((4, 4, 4, 4), dtype='float32')
# layer = BatchRN(r_max=3., d_max=5.)
# print(layer.dtype)  # float32.
# y = layer(x)  # MyLayer will not cast inputs to it's dtype of float32
# print(y.dtype)  # float64
