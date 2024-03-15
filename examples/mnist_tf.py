import itertools
import os
import sys
import time

import tensorflow as tf
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../bitnet"))

from bitnet_tf import BitLinear

kl = tf.keras.layers

# use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []
        self.total_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_t0)

    def on_train_end(self, logs=None):
        self.total_times = list(itertools.accumulate(self.times))


def _train_mnist(
    model: tf.keras.Model,
    epochs: int,
    batch_size: int,
    lr: float,
):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer, loss_fn, metrics=["accuracy"])

    time_callback = TimeHistory()
    history = model.fit(
        x_train,
        y_train,
        batch_size,
        epochs,
        validation_data=[x_test, y_test],
        callbacks=[time_callback],
    )
    score = model.evaluate(x_test, y_test)

    return history.history, time_callback.total_times, score


def simple_test():
    model = tf.keras.models.Sequential(
        [
            kl.Flatten(input_shape=(28, 28)),
            BitLinear(128),
            kl.Activation("relu"),
            kl.Dense(10),
        ]
    )
    model.summary()
    _, _, score = _train_mnist(model, 5, 512, 0.001)
    print("Test loss:", score[0])  # Test loss: 0.12200380861759186
    print("Test acc :", score[1])  # Test acc : 0.9624999761581421


def compare(units: int, layers: int, lr: float, epochs: int):
    models = []

    # --- dense
    m = tf.keras.models.Sequential()
    m.add(kl.Flatten(input_shape=(28, 28)))
    for _ in range(layers):
        m.add(kl.LayerNormalization())
        m.add(kl.Dense(units, use_bias=False))
        m.add(kl.Activation("relu"))
    m.add(kl.Dense(10))
    models.append(["Dense", m])

    # --- 1bit
    m = tf.keras.models.Sequential()
    m.add(kl.Flatten(input_shape=(28, 28)))
    for _ in range(layers):
        m.add(BitLinear(units, "1bit", flg_before_linear=False))
        m.add(kl.Activation("relu"))
    m.add(kl.Dense(10))
    models.append(["BitLinear 1bit", m])

    # --- 1.58bit
    m = tf.keras.models.Sequential()
    m.add(kl.Flatten(input_shape=(28, 28)))
    for _ in range(layers):
        m.add(BitLinear(units, "1.58bit"))
        m.add(kl.Activation("relu"))
    m.add(kl.Dense(10))
    models.append(["BitLinear 1.58bit", m])

    for name, m in models:
        history, times, _ = _train_mnist(m, epochs, 512, lr)
        plt.plot(times, history["val_accuracy"], label=name)

    plt.ylim(0, 1)
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    simple_test()
    # compare(units=64, layers=5, lr=0.0001, epochs=20)
