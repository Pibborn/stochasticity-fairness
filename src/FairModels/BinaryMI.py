from __future__ import annotations
from typing import Any, Union

import gc, six
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

from FairModels.BaseModel import *


class BinaryMI(BaseModel):
    """
        Stochastically Quantized Neural Network which has a bernoulli
        activation after each layer to exactly compute the mutual information.
    """

    def __init__(self,
                # BinaryMI HPs
                num_hidden_layers: int = 0,
                size_hidden_layers: int = 0,
                hidden_layers: list[int] = [1024, 512, 128],
                batch_normalisation_layers: list[int] = [1024, 512, 128],
                quantized_position: list[bool] = [False, True, False],
                set_quantized_position: bool = False,
                batch_normalisation: bool = False,
                activation_binary: str = 'bernoulli',
                activation_nonbinary: str = 'tanh',
                kernel_regularizer: float = 0.01,
                drop_out: float = 0.,
                gamma: float = 0.0,
                # Common HPs
                batch_size: int = 200,
                learning_rate: float = 0.001,
                learning_rate_decay_rate: float = 1,
                learning_rate_decay_steps: int = 1000,
                optimizer: str = "Adam",
                epoch: int = 10,
                loss: str = 'binary_crossentropy',
                run_eagerly: bool = False,
                # other variables
                verbose: int = 0,
                validation_size: float = 0.1,
                input_shape: Union[tuple, int] = 0,
                last_layer_size: int = 2,
                random_seed: int = 42,
                name: str = "BinaryMI",
                dataset_name: str = "Mnist",
                checkpoint_path: str = "",
                datetime: str = "",
                conv: bool = False
        ) -> None:
        super().__init__(
            # DirectRankerAdv HPs
            num_hidden_layers=num_hidden_layers,
            size_hidden_layers=size_hidden_layers,
            hidden_layers=hidden_layers,
            batch_normalisation_layers=batch_normalisation_layers,
            quantized_position=quantized_position,
            set_quantized_position=set_quantized_position,
            batch_normalisation=batch_normalisation,
            activation_binary=activation_binary,
            activation_nonbinary=activation_nonbinary,
            kernel_regularizer=kernel_regularizer,
            drop_out=drop_out,
            # Common HPs
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_decay_rate=learning_rate_decay_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss,
            # other variables
            verbose=verbose,
            validation_size=validation_size,
            random_seed=random_seed,
            name=name,
            dataset_name=dataset_name,
            checkpoint_path=checkpoint_path,
            datetime=datetime,
            run_eagerly=run_eagerly,
            last_layer_size=last_layer_size,
            conv=conv
        )

        self.input_shape = input_shape
        self.x_input = Any
        self.out = Any
        self.gamma = gamma

    def mutual_information_bernoulli_loss(self, y_true, y_pred):
        """
        I(x;y)  = H(x)   - H(x|y)
                = H(L_n) - H(L_n|s)
                = H(L_n) - (H(L_n|s=0) + H(L_n|s=1))
        H_bernoulli(x) = -(1-theta) x ln(1-theta) - theta x ln(theta)
        here theta => probability for 1 and 1-theta => probability for 0

        pseudocode:
        def get_h_bernoulli(l):
            theta = np.mean(l, axis=0)
            return -(1-theta) * np.log2(1-theta) - theta * np.log2(theta)

        y_pred = np.random.binomial(n=1, p=0.6, size=[2000, 5])
        y_true = np.random.binomial(n=1, p=0.6, size=[2000])

        y_pred[y_true == 0] = np.random.binomial(n=1, p=0.5, size=[len(y_true[y_true == 0]), 5])
        y_pred[y_true == 1] = np.random.binomial(n=1, p=0.8, size=[len(y_true[y_true == 1]), 5])

        H_L_n = get_h_bernoulli(y_pred)
        H_L_n_s0 = get_h_bernoulli(y_pred[y_true == 0])
        H_L_n_s1 = get_h_bernoulli(y_pred[y_true == 1])

        counts = np.bincount(y_true)

        MI = H_L_n - ((counts[0] / 2000 * H_L_n_s0) + (counts[1] / 2000 * H_L_n_s1))

        return np.sum(MI)

        :param y_pred: output of the layer
        :param y_true: sensitive attribute
        :return: The loss
        """

        def get_theta(x):
            alpha = None
            temperature = 6.0
            use_real_sigmoid = True
            # hard_sigmoid
            _sigmoid = tf.keras.backend.clip(0.5 * x + 0.5, 0.0, 1.0)
            if isinstance(alpha, six.string_types):
                assert self.alpha in ["auto", "auto_po2"]

            if isinstance(alpha, six.string_types):
                len_axis = len(x.shape)

                if len_axis > 1:
                    if K.image_data_format() == "channels_last":
                        axis = list(range(len_axis - 1))
                    else:
                        axis = list(range(1, len_axis))
                else:
                    axis = [0]

                std = K.std(x, axis=axis, keepdims=True) + K.epsilon()
            else:
                std = 1.0

            if use_real_sigmoid:
                p = tf.keras.backend.sigmoid(temperature * x / std)
            else:
                p = _sigmoid(temperature * x / std)

            return p

        def log2(x):
            numerator = tf.math.log(x + 1e-20)
            denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
            return numerator / denominator

        def get_h_bernoulli(tensor):
            theta = tf.reduce_mean(get_theta(tensor), axis=0)
            return tf.reduce_sum(-(1 - theta) * log2(1 - theta) - theta * log2(theta))

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float64)
        num_classes = 2
        H_L_n = get_h_bernoulli(y_pred)
        H_L_n_s = []
        norm_s = []
        for i in range(num_classes):
            if tf.shape(y_true).shape[0] == 1:
                y_filter = tf.where(y_true == i)
            else:
                y_filter = tf.where(y_true[:, 0] == i)[:, 0]

            y_i = tf.gather(y_pred, indices=y_filter)
            H_L_n_si = get_h_bernoulli(y_i)
            H_L_n_s.append(H_L_n_si)
            cnt_i = tf.shape(y_i)[0] + tf.cast(1e-16, dtype=tf.int32)  # number of repr with index i
            norm_si = cnt_i / tf.shape(y_pred)[0]
            norm_s.append(norm_si)

        norm_s = tf.convert_to_tensor(norm_s)

        H_L_n_s = tf.convert_to_tensor(H_L_n_s)
        MI = H_L_n - tf.reduce_sum(tf.math.multiply(norm_s, H_L_n_s))

        # NOTE: this is a hotfix when we dont have all classes
        MI = tf.where(tf.math.is_nan(MI), tf.convert_to_tensor([0.0], dtype=tf.float64), MI)
        return MI

    def _build_model(self) -> None:
        r"""
            Is building the model by sequential adding layers.

            1. The data (x) gets into an input layer which is activated
            with a bernoulli layer via a sigmoid:

            .. math::
                \hat{x} = \mathrm{Bern} (\sigma(6.0 \cdot x / 1.0))

            2. The size of the next layers are defined with self.hidden_layers.
            The basic architecture is build via:

            .. code-block:: python

                for i in self.hidden_layers:
                    layer = Dense()(layer)
                    layer = QActivation("bernoulli")(layer)

            3. To avoid overfitting it is often useful to lower the learning rate
            during the training. Therefore, the model is build using a schedule
            which applies the inverse decay function to an initial learning rate.

            4. Different optimizers can be used such as Adam, Nadam or SGD.

            Args:
                NoInput
            Returns:
                None
        """

        # placeholders for the inputs: shape depends on whether we have a convnet or not
        self.x_input = tf.keras.layers.Input(
            shape=self.input_shape,
            name="x"
        )

        # build layers layers
        self.out, self.last_quantized = self._get_hidden_qlayer(
            self.x_input,
            hidden_layer=self.hidden_layers,
            drop_out=self.drop_out,
            name="t",
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
            conv=self.conv
        )

        if sum(self.quantized_position) > 0:
            outputs = [self.out, self.last_quantized]
            # we assume that we only have one quantized layer here
            if sum(self.quantized_position) > 1:
                raise ValueError('Only one binary layer is allowed')
            index_qact = [i for i, x in enumerate(self.quantized_position) if x][-1]
            loss = {
                f"t_{len(self.hidden_layers)}": self.loss,
                f"t_{index_qact}": self.mutual_information_bernoulli_loss
            }
            lossWeights = {f"t_{len(self.hidden_layers)}": float(1 - self.gamma), f"t_{index_qact}": float(self.gamma)}
            metrics = {f"t_{len(self.hidden_layers)}": 'AUC' if self.last_layer_size == 1 else 'acc', f"t_{index_qact}": 'acc'}
        else:
            outputs = self.out
            loss = [self.loss]
            lossWeights = float(1)
            metrics = ['AUC'] if self.last_layer_size == 1 else ['acc']

        # create the model
        self.model = tf.keras.models.Model(
            inputs=self.x_input,
            outputs=outputs,
            name=self.name
        )

        if self.verbose > 1:
            self.model.summary() # type: ignore
            self._plot_model(self.model, "binaryMI.png")

        # setup learning rate schedule
        # TODO: maybe we have to go with a simple learning rate here for the experiments
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=self.learning_rate_decay_steps,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False
        )

        if sum(self.quantized_position) > 0:
            self.model.compile( # type: ignore
                optimizer=self.optimizer(lr_schedule),
                loss=loss,
                loss_weights=lossWeights,
                metrics=metrics,
                run_eagerly=self.run_eagerly
            )
        else:
            self.model.compile( # type: ignore
                optimizer=self.optimizer(lr_schedule),
                loss=loss,
                metrics=metrics,
                run_eagerly=self.run_eagerly
            )

    def fit(self, x: np.ndarray, y: np.ndarray, s: np.ndarray, **fit_params: dict[Any, Any]) -> None:
        """
            Fit function which first build the model and than
            fits it using x_train and y_train. A special callback
            class is used to compute the mutual information at the
            beginning of each epoch and at the end of the training.

            Args:
                NDArray: x_train
                NDArray: y_train
                NDArray: s_train
                dict[Any]: fit_params

            Returns:
                None
        """
       # set seed
        tf.random.set_seed(self.random_seed)

        # get the correct numpy type
        x = self.convert_array_to_float(x)
        y = self.convert_array_to_float(y)
        s = self.convert_array_to_float(s)

        # input size
        self.input_shape = (x.shape[1], )

        # convert for classifier output
        y = tf.keras.utils.to_categorical(y, self.last_layer_size)

        self._build_model()

        self.history = self.model.fit( # type: ignore
            x=x,
            y=[y, s],
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=self.verbose,
            shuffle=True,
            validation_split=self.validation_size
        ).history
        # https://github.com/tensorflow/tensorflow/issues/14181
        # https://github.com/tensorflow/tensorflow/issues/30324
        gc.collect()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
            Get the class probablities.

            Args:
                NDArray: features
            Returns:
                NDArray: class probablities
        """
        # get the correct numpy type
        features = self.convert_array_to_float(features)

        if len(features.shape) == 1:
            features = [features] # type: ignore

        res = self.model.predict( # type: ignore
            features,
            batch_size=self.batch_size,
            verbose=str(self.verbose)
        )[0]

        if self.last_layer_size > 1:
            return res[:,1]
        return res

    def get_representations(self, x, split=False):
        # get the correct numpy type
        x = self.convert_array_to_float(x)

        # get binary layer
        return self.model(x)[1].numpy()
