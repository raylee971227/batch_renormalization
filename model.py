import tensorflow as tf
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar100
from keras import regularizers, optimizers
from keras.utils import multi_gpu_model
# from renorm import BatchRenorm
from renorm2 import BatchRenormalization
import numpy as np

LAM = 0.001
NUM_CLASSES = 100
BATCH_SIZE = 64
LAM = 1e-4


class Data(object):
    def __init__(self, number_classes=NUM_CLASSES):
        """
         Returns the CIFAR Dataset for Training, Evaluation, and Testing.
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()
        self.x_train = self.format_data(self.x_train)
        self.x_test = self.format_data(self.x_test)

        self.y_train = np_utils.to_categorical(self.y_train, number_classes)
        self.y_test = np_utils.to_categorical(self.y_test, number_classes)

        self.x_train, self.y_train, self.x_eval, self.y_eval = self.get_val_set(1 / 5)
        print(
            self.x_train.shape[0],
            self.y_train.shape[0],
            self.x_eval.shape[0],
            self.y_eval.shape[0],
        )

    @staticmethod
    def format_data(x):
        """
        Format the data for tensorflow
        """
        x = x.astype("float32")
        return x

    def get_val_set(self, size):
        """
        Get a validation set
        """
        index = np.random.choice(
            np.arange(self.x_train.shape[0]), self.x_train.shape[0]
        )
        thresh = int(self.x_train.shape[0] * size)
        training_idx, test_idx = index[:thresh], index[thresh:]
        x_eval, x_train = self.x_train[training_idx, :], self.x_train[test_idx, :]
        y_eval, y_train = self.y_train[training_idx, :], self.y_train[test_idx, :]
        return x_train, y_train, x_eval, y_eval


class Model(tf.Module):
    def __init__(self, number_classes=NUM_CLASSES, lamda=LAM):
        """
        A Convolutional neural net
        """

        self.model = Sequential()
        # self.model.add(BatchNormalization(input_shape=(32, 32, 3)))
        # self.model.add(BatchRenorm())
        self.model.add(
            Conv2D(
                32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(lamda), input_shape=(32, 32, 3)
            )
        )
        self.model.add(Activation("relu"))
        self.model.add(BatchRenormalization())
        # self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(
                32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(lamda)
            )
        )
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(
            Conv2D(
                64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(lamda)
            )
        )
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(
                64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(lamda)
            )
        )
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.4))

        self.model.add(
            Conv2D(
                128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(lamda)
            )
        )
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(
                128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(lamda)
            )
        )
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(number_classes, activation="softmax"))

    def train(
        self,
        x_train,
        y_train,
        x_eval,
        y_eval,
        learn_rate,
        num_epoch,x
        batch_size=BATCH_SIZE,
    ):
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
        )
        datagen.fit(data.x_train)

        opt_rms = keras.optimizers.rmsprop(lr=learn_rate, decay=1e-6)
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=opt_rms,
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )
        self.model.summary()
        self.model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=num_epoch,
            verbose=1,
            # validation_data=(x_eval, y_eval),
        )

    def evaluate(self, x_test, y_test, batch_size=BATCH_SIZE):
        return self.model.evaluate(x_test, y_test, verbose=1)


if __name__ == "__main__":
    data = Data()
    model = Model()

    learning_rates = [0.001, 0.0005, 0.0003, 0.0001, 0.00005, 0.00003, 0.00001]

    for rate in learning_rates:
        if rate == 0.001:
            epoch = 75
        else:
            epoch = 25

        model.train(
            data.x_train,
            data.y_train,
            data.x_eval,
            data.y_eval,
            learn_rate=rate,
            num_epoch=epoch,
        )

    # model.train(data.x_train, data.y_train, data.x_eval, data.y_eval, learn_rate=0.001, num_epoch=75)
    # model.train(data.x_train, data.y_train, data.x_eval, data.y_eval, learn_rate=0.0005, num_epoch=25)
    # model.train(data.x_train, data.y_train, data.x_eval, data.y_eval, learn_rate=0.0003, num_epoch=25)
    # model.train(data.x_train, data.y_train, data.x_eval, data.y_eval, learn_rate=0.0001, num_epoch=25)
    # model.train(data.x_train, data.y_train, data.x_eval, data.y_eval, learn_rate=0.00005, num_epoch=25)
    # model.train(data.x_train, data.y_train, data.x_eval, data.y_eval, learn_rate=0.00003, num_epoch=25)
    # model.train(data.x_train, data.y_train, data.x_eval, data.y_eval, learn_rate=0.00001, num_epoch=25)

    scores = model.evaluate(data.x_test, data.y_test)
    print("\nTest result: %.3f loss: %.3f" % (scores[1] * 100, scores[0]))
    print(scores)
