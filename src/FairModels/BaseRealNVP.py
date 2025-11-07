import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import regularizers
from tensorflow.keras import layers

import numpy as np

class ScaleLayer(tf.keras.layers.Layer):

    def __init__(self, n: int, name: str):
        super(ScaleLayer, self).__init__()
        self.scale = tf.Variable([1.] * n, name=name)
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs * self.scale

class BaseRealNVP(keras.Model):

    def __init__(self, 
        num_features: int, 
        num_coupling_layers: int = 6,
        t_layers: list[int] = [], 
        s_layers: list[int] = [],
        t_activation: str = "relu", 
        s_activation: str = "relu", 
        t_last_activation: str = "linear",
        s_last_activation: str = "tanh",
        rescale_s: bool = True,
        reg: float = 0.01,
        std: float = 1.0, 
        mean: float = 0.0,
        name: str = "NVP"
        ):

        super(BaseRealNVP, self).__init__(name=name)

        self.num_features = num_features
        self.num_coupling_layers = num_coupling_layers
        self.t_layers = t_layers
        self.s_layers = s_layers
        self.t_activation = t_activation
        self.s_activation = s_activation
        self.t_last_activation = t_last_activation
        self.s_last_activation = s_last_activation
        self.rescale_s = rescale_s
        self.reg = reg
        self.std = std
        self.mean = mean

        self._build_model()

    def coupling(self, num):
        
        input_layer = keras.layers.Input(
            shape=(self.t_layers[-1], ),
            name="{}_input_{}".format(num, self.name)
            )

        t_layer = keras.layers.Dense(
            self.t_layers[0], 
            activation=self.t_activation, 
            kernel_regularizer=regularizers.l2(self.reg),
            name="t{}_layer_0_{}".format(num, self.name)
        )(input_layer)

        for i, tl in enumerate(self.t_layers[1:-1]):
            t_layer = keras.layers.Dense(
                tl, 
                activation=self.t_activation, 
                kernel_regularizer=regularizers.l2(self.reg),
                name="t{}_{}_layer_1_{}".format(num, i, self.name)
            )(t_layer)

        t_layer = keras.layers.Dense(
            self.t_layers[-1], 
            activation=self.t_last_activation, 
            kernel_regularizer=regularizers.l2(self.reg),
            name="t{}_layer_2_{}".format(num, self.name)
        )(t_layer)

        s_layer = keras.layers.Dense(
            self.s_layers[0], 
            activation=self.s_activation, 
            kernel_regularizer=regularizers.l2(self.reg),
            name="s{}_layer_0_{}".format(num, self.name)
        )(input_layer)

        for i, sl in enumerate(self.s_layers[1:-1]):
            s_layer = keras.layers.Dense(
                sl, 
                activation=self.s_activation, 
                kernel_regularizer=regularizers.l2(self.reg),
                name="s{}_{}_layer_1_{}".format(num, i, self.name)
            )(s_layer)

        s_layer = keras.layers.Dense(
            self.s_layers[-1], 
            activation=self.s_last_activation, 
            kernel_regularizer=regularizers.l2(self.reg),
            name="s{}_layer_2_{}".format(num, self.name)
        )(s_layer)

        if self.rescale_s:
            s_layer = ScaleLayer(self.s_layers[-1], name="rescale_{}".format(self.name))(s_layer)

        return keras.Model(inputs=input_layer, outputs=[s_layer, t_layer])

    def _build_model(self):

        # TODO: this could fail in the future since its hard coded
        num_features = self.num_features[0]

        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[self.mean]*num_features, scale_diag=[self.std]*num_features
        )

        # Last layer in t_layers and s_layers has to be of size `dimension`
        if not self.t_layers:
            self.t_layers = [num_features]
        elif self.t_layers[-1] != num_features:
            self.t_layers.append(num_features)
        if not self.s_layers:
            self.s_layers = [num_features]
        elif self.s_layers[-1] != num_features:
            self.s_layers.append(num_features)

        self.masks = np.array([[0 if (i+j)%2==0 else 1 for i in range(num_features)]
                                    for j in range(self.num_coupling_layers)], dtype="float32")

        self.layers_list = [self.coupling(num=i) for i in range(self.num_coupling_layers)]

    def call(self, x, training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = tf.convert_to_tensor(
                tf.cast(x, tf.float32) * tf.cast(self.masks[i], tf.float32), 
                dtype=tf.float32
                )
            reversed_mask = tf.convert_to_tensor(1 - self.masks[i], dtype=tf.float32)
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                tf.cast(reversed_mask, tf.float32)
                * (tf.cast(x, tf.float32) * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return x, log_det_inv

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet

        return -tf.reduce_mean(log_likelihood)
