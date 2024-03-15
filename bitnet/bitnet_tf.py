import numpy as np
import tensorflow as tf
from tensorflow import keras


class BitLinear(keras.layers.Layer):
    def __init__(
        self,
        units: int,
        weight_type: str = "1.58bit",  # "1bit" or "1.58bit"
        bits: int = 8,
        use_bias: bool = False,
        flg_before_linear: bool = True,
    ):
        super().__init__()
        self.units = units
        self.weight_type = weight_type
        self.bits = bits
        self.use_bias = use_bias
        self.flg_before_linear = flg_before_linear

        self.Qb = 2 ** (bits - 1)
        self.eps = 1e-5
        self._dtype = tf.float32

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.weight = self.add_weight(
            name="weight",
            shape=[last_dim, self.units],
            dtype=self._dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.units],
                dtype=self._dtype,
                trainable=True,
            )
        else:
            self.bias = None

        self.norm = keras.layers.LayerNormalization()
        self.built = True

    def call(self, inputs):
        x = self.norm(inputs)

        # --- 1. quantized weights
        beta = tf.reduce_mean(tf.abs(self.weight))
        if self.weight_type == "1bit":
            # binarized quantize
            alpha = tf.reduce_mean(self.weight)
            weight = self.ste_sign(self.weight - alpha)
        elif self.weight_type == "1.58bit":
            # absmean quantization
            weight = self.weight / (beta + self.eps)
            weight = self.ste_round(weight)
            weight = tf.clip_by_value(weight, -1, 1)
        else:
            raise ValueError(self.weight_type)

        # --- 2. quantized inputs, absmax quantization
        gamma = tf.reduce_max(tf.abs(x))
        if self.flg_before_linear:
            # [-Qb, Qb]
            x = x * self.Qb / gamma
            x = tf.clip_by_value(x, -self.Qb + self.eps, self.Qb - self.eps)
        else:
            # [0, Qb]
            eta = tf.reduce_min(x)
            x = (x - eta) * self.Qb / gamma
            x = tf.clip_by_value(x, self.eps, self.Qb - self.eps)

        # --- 3. calc
        # Addition is faster than multiplication, but there is no implementation
        x = tf.matmul(x, weight)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)

        # --- 4. dequantized inputs
        x = x * gamma * beta / self.Qb

        return x

    def ste_sign(self, x):
        # tf.sign(0) -> 0 so I made it myself
        x2 = tf.cast(tf.where(x > 0, 1, -1), x.dtype)
        return tf.stop_gradient(x2 - x) + x  # STE

    def ste_round(self, x):
        x2 = tf.round(x)
        return tf.stop_gradient(x2 - x) + x  # STE


def _check_grad():

    # --- sign(where)
    x = tf.constant([0.9, 0, -1])
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.where(x > 0, 1, -1)
    grads = tape.gradient(y, x)
    print("sign(where):", grads, y)

    # --- round
    x = tf.constant([0.9, 1.2, 2.5])
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.round(x)
    grads = tape.gradient(y, x)
    print("round:", grads, y)

    # --- clip_by_value
    x = tf.constant([0.9, 1.2, 2.5])
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.clip_by_value(x, -1, 1)
    grads = tape.gradient(y, x)
    print("clip_by_value:", grads, y)


def _main():
    x = np.array([[1, 2], [1, 1]], np.float32)
    m = BitLinear(32)
    y = m(x)
    print(y)


if __name__ == "__main__":
    _check_grad()
    _main()
