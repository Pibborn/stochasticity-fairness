import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers

import gc
from FairModels.BaseRealNVP import BaseRealNVP
from FairModels.BaseModel import *

from sklearn.model_selection import train_test_split

import numpy as np


class FairRealNVPCls(keras.Model, BaseModel):

    def __init__(self,
            # NVP HPs
            num_coupling_layers_a=6,
            num_coupling_layers_b=6,
            num_t_layers_a=4,
            num_s_layers_a=4,
            size_t_layers_a=256,
            size_s_layers_a=256,
            num_t_layers_b=4,
            num_s_layers_b=4,
            size_t_layers_b=256,
            size_s_layers_b=256,
            t_activation_a="relu",
            s_activation_a="relu",
            t_activation_b="relu",
            s_activation_b="relu",
            t_last_activation_a="linear",
            s_last_activation_a="tanh",
            t_last_activation_b="linear",
            s_last_activation_b="tanh",
            rescale_s_a=True,
            rescale_s_b=True,
            # dr HPs
            num_hidden_layers=2,
            size_hidden_layers=20,
            feature_activation='tanh',
            drop_out=0,
            # common HPs
            gamma=1.,
            z0_loss_gamma=1,
            learning_rate=0.001,
            learning_rate_decay_rate=1,
            learning_rate_decay_steps=1000,
            optimizer="Adam",
            reg=0.01,
            epoch=100,
            verbose=0,
            validation_split=0.0,
            std=1.0,
            mean=0.0,
            batch_size=256,
            dataset_name="Compas",
            name="FairRealNVP",
            replace_s=False,
            nan_loss=False,
            run_eagerly=False,
            random_seed=42,
            interpretable=False,
            use_norm=True,
            z0_zero=True,
            z0_loss=True
            ):

        super(FairRealNVPCls, self).__init__(name=name)

        self.num_coupling_layers_a = num_coupling_layers_a
        self.num_coupling_layers_b = num_coupling_layers_b
        self.t_layers_a = [size_t_layers_a for i in range(num_t_layers_a)]
        self.s_layers_a = [size_s_layers_a for i in range(num_s_layers_a)]
        self.t_layers_b = [size_t_layers_b for i in range(num_t_layers_b)]
        self.s_layers_b = [size_s_layers_b for i in range(num_s_layers_b)]
        self.size_t_layers_a = size_t_layers_a
        self.size_s_layers_a = size_s_layers_a
        self.size_t_layers_b = size_t_layers_b
        self.size_s_layers_b = size_s_layers_b
        self.num_t_layers_a = num_t_layers_a
        self.num_s_layers_a = num_s_layers_a
        self.num_t_layers_b = num_t_layers_b
        self.num_s_layers_b = num_s_layers_b
        self.t_activation_a = t_activation_a
        self.s_activation_a = s_activation_a
        self.t_activation_b = t_activation_b
        self.s_activation_b = s_activation_b
        self.t_last_activation_a = t_last_activation_a
        self.s_last_activation_a = s_last_activation_a
        self.t_last_activation_b = t_last_activation_b
        self.s_last_activation_b = s_last_activation_b
        self.rescale_s_a = rescale_s_a
        self.rescale_s_b = rescale_s_b
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.hidden_layers = [size_hidden_layers for i in range(num_hidden_layers)]
        self.feature_activation = feature_activation
        self.drop_out = drop_out
        self.gamma = gamma
        self.z0_loss_gamma = z0_loss_gamma
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        if optimizer == "Adam":
            self.optimizer = tf.keras.optimizers.Adam
        self.optimizer_name = optimizer
        self.reg = reg
        self.epoch = epoch
        self.verbose = verbose
        self.validation_split = validation_split
        self.std = std
        self.mean = mean
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.replace_s = replace_s
        self.nan_loss = nan_loss
        self.random_seed = random_seed
        self.interpretable = interpretable
        self.use_norm = use_norm
        self.z0_zero = z0_zero
        self.z0_loss = z0_loss
        self.run_eagerly = run_eagerly

        self.loss_tracker_a = keras.metrics.Mean(name="loss_a")
        self.loss_tracker_a_s = keras.metrics.Mean(name="loss_a_s")
        self.loss_tracker_b = keras.metrics.Mean(name="loss_b")
        self.loss_tracker_cls = keras.metrics.Mean(name="loss_cls")

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker_a, self.loss_tracker_a_s, self.loss_tracker_b, self.loss_tracker_cls]

    def _build_model(self):

        self.model_a = BaseRealNVP(
            num_features=self.num_features,
            num_coupling_layers=self.num_coupling_layers_a,
            t_layers=self.t_layers_a,
            s_layers=self.s_layers_a,
            t_activation=self.t_activation_a,
            s_activation=self.s_activation_a,
            t_last_activation=self.t_last_activation_a,
            s_last_activation=self.s_last_activation_a,
            rescale_s=self.rescale_s_a,
            reg=self.reg,
            std=self.std,
            mean=self.mean,
            name="NVP_A"
        )

        self.model_b = BaseRealNVP(
            num_features=self.num_features,
            num_coupling_layers=self.num_coupling_layers_b,
            t_layers=self.t_layers_b,
            s_layers=self.s_layers_b,
            t_activation=self.t_activation_b,
            s_activation=self.s_activation_b,
            t_last_activation=self.t_last_activation_b,
            s_last_activation=self.s_last_activation_b,
            rescale_s=self.rescale_s_b,
            reg=self.reg,
            std=self.std,
            mean=self.mean,
            name="NVP_B"
        )

        self.cls = self.cls_layer()

        self.lossWeights = {
            "loss_a": self.gamma,
            "loss_b": self.gamma,
            "loss_cls": 1.0 - self.gamma
        }

        if self.z0_loss:
            self.lossWeights["loss_a_s"] = self.gamma

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=self.learning_rate_decay_steps,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False
        )

        self.compile(
            optimizer=self.optimizer(lr_schedule),
            run_eagerly=self.run_eagerly
        )

        if self.verbose > 1:
            self.cls.summary()

    def cls_layer(self):

        input_layer = layers.Input(shape=self.num_features, name="input_cls")
        x = layers.Input(shape=self.num_features, name="x")

        nn = layers.Dense(
            units=self.hidden_layers[0],
            activation=self.feature_activation,
            kernel_regularizer=tf.keras.regularizers.l2(self.reg),
            bias_regularizer=tf.keras.regularizers.l2(self.reg),
            activity_regularizer=tf.keras.regularizers.l2(self.reg),
            name="nn_0_ranking"
        )(input_layer)

        if self.drop_out > 0:
            nn = layers.Dropout(self.drop_out)(nn)

        for i in range(1, len(self.hidden_layers)):
            nn = layers.Dense(
                units=self.hidden_layers[i],
                activation=self.feature_activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.reg),
                bias_regularizer=tf.keras.regularizers.l2(self.reg),
                activity_regularizer=tf.keras.regularizers.l2(self.reg),
                name="nn_{}_ranking".format(i)
            )(nn)

            if self.drop_out > 0:
                nn = layers.Dropout(self.drop_out)(nn)

        feature_part = keras.Model(input_layer, nn, name='feature_part')

        nn = feature_part(x)

        out = layers.Dense(
            units=1,
            activation="sigmoid",
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(self.reg),
            activity_regularizer=tf.keras.regularizers.l2(self.reg),
            name="cls_part"
        )(nn)

        return keras.Model(inputs=[x], outputs=out, name="cls")

    def call_cls(self, x, training=True):
        x = self(x, training=False)
        return self.cls(x, training=training)

    def call(self, x, training=True):
        if training:
            z, _ = self.model_a(x[0])
        else:
            z, _ = self.model_a(x)
            if self.z0_zero:
                z_0 = z[:,0]
                z_0 *= 0
                z_0 = tf.reshape(z_0, [tf.shape(z_0)[0], 1])
                z = tf.concat([z_0, z[:,1:]], 1)
        x, _ = self.model_b(z, training=False)
        return x

    def loss_cls(self, x, y):
        pred = self.call_cls(x)
        return tf.keras.losses.MeanSquaredError()(y, pred)

    def train_var(self, model=""):
        train_var = []
        for var in self.trainable_variables:
            for m in model:
                if m in var.path:
                    train_var.append(var)
        return train_var

    def train_step(self, data):

        # data = [x, [y, s]]
        data = [data[0], data[1][:,0], data[1][:,1]]
        # get x values for nvp a
        data_a = data[0]
        # get x values for nvp b = x[s==smax]
        data_b = data[0][data[2] == self.s_max]

        with tf.GradientTape() as tape:
            loss_a = self.model_a.log_loss(data_a) * self.lossWeights["loss_a"]
            gradients = tape.gradient(loss_a, self.train_var(model=["NVP_A"]))
            train_var = self.train_var(model=["NVP_A"])

        if self.z0_loss:
            with tf.GradientTape() as tape:
                model_a_out = self.model_a(data_a)[0][:,0]
                model_a_out = (model_a_out - tf.reduce_min(model_a_out)) / (tf.reduce_max(model_a_out) - tf.reduce_min(model_a_out))
                loss_a_s = tf.keras.losses.binary_crossentropy(data[2], model_a_out) * self.lossWeights["loss_a_s"]
                gradients.extend(tape.gradient(loss_a_s, self.train_var(model=["NVP_A"])))
                train_var.extend(self.train_var(model=["NVP_A"]))

        with tf.GradientTape() as tape:
            loss_b = self.model_b.log_loss(data_b) * self.lossWeights["loss_b"]
            gradients.extend(tape.gradient(loss_b, self.train_var(model=["NVP_B"])))
            train_var.extend(self.train_var(model=["NVP_B"]))

        with tf.GradientTape() as tape:
            # data for ranker: x0, x1 and y
            loss_cls = self.loss_cls(data[0], data[1]) * self.lossWeights["loss_cls"]
            gradients.extend(tape.gradient(loss_cls, self.trainable_variables))
            train_var.extend(self.trainable_variables)

        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, train_var))

        self.loss_tracker_a.update_state(loss_a)
        if self.z0_loss:
            self.loss_tracker_a_s.update_state(loss_a_s)
        self.loss_tracker_b.update_state(loss_b)

        # TODO: this can be a hack to handle the NaN loss
        # value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(loss_cls)), dtype=tf.float32)
        self.loss_tracker_cls.update_state(loss_cls)

        loss_dict = {
            "loss_a": self.loss_tracker_a.result(),
            "loss_b": self.loss_tracker_b.result(),
            "loss_cls": self.loss_tracker_cls.result()
        }

        if self.z0_loss:
            loss_dict["loss_a_s"] = self.loss_tracker_a_s.result()

        return loss_dict

    def test_step(self, data):

        # data = [x, [y, s]]
        data = [data[0], data[1][:,0], data[1][:,1]]
        # get x values for nvp a
        data_a = data[0]
        # get x values for nvp b = x[s==smax]
        data_b = data[0][data[2] == self.s_max]

        loss_a = self.model_a.log_loss(data_a) * self.lossWeights["loss_a"]
        self.loss_tracker_a.update_state(loss_a)

        if self.z0_loss:
            model_a_out = self.model_a(data_a)[0][:,0]
            model_a_out = (model_a_out - tf.reduce_min(model_a_out)) / (tf.reduce_max(model_a_out) - tf.reduce_min(model_a_out))
            loss_a_s = tf.keras.losses.binary_crossentropy(data[2], model_a_out) * self.lossWeights["loss_a_s"]
            self.loss_tracker_a_s.update_state(loss_a_s)

        loss_b = self.model_b.log_loss(data_b) * self.lossWeights["loss_b"]
        self.loss_tracker_b.update_state(loss_b)

        loss_cls = self.loss_cls(data[0], data[1]) * self.lossWeights["loss_cls"]
        self.loss_tracker_cls.update_state(loss_cls)

        loss_dict = {
            "loss_a": self.loss_tracker_a.result(),
            "loss_b": self.loss_tracker_b.result(),
            "loss_cls": self.loss_tracker_cls.result()
        }

        if self.z0_loss:
            loss_dict["loss_a_s"] = self.loss_tracker_a_s.result()

        return loss_dict

    def norm_data(self, data):
        if self.use_norm:
            return (data - np.max(data)) / (np.max(data) - np.min(data))
        else:
            return data

    def fit(self, x, y, s):

        if len(s.shape) == 1:
            s = np.array([[si] for si in s])

        if x[s.flatten() == 1].shape[0] > x[s.flatten() == 0].shape[0]:
            self.s_max = 1
            self.s_min = 0
        else:
            self.s_max = 0
            self.s_min = 1

        self.num_features = (x.shape[1], )
        self._build_model()

        if len(y.shape) == 1:
            y = np.array([[yi] for yi in y])

        # get the correct numpy type
        x = self.convert_array_to_float(x)
        y = self.convert_array_to_float(y)
        s = self.convert_array_to_float(s)

        # NaN value callback -- stops training
        loss_name = ["loss_a", "loss_b", "loss_cls"]
        if self.z0_loss: loss_name.append("loss_a_s")
        callbacks = [TerminateOnNaN(loss_name=loss_name, verbose=self.verbose)]

        # early stopping callback -- restors the weight when we have nan loss
        # NOTE: this is not the best way to do this since in the case of a normal
        # training this will trigger as well
        stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss_cls',
            min_delta=0,
            patience=3,
            verbose=self.verbose,
            mode='min',
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0
        )
        callbacks.append(stopping)
        self.nan_loss = callbacks[0].nan_loss

        if self.validation_split > 0:
            x, X_test, y_train, y_test, S_train, S_test = \
                            train_test_split(x, y, s, test_size=self.validation_split)
            callbacks.append(CallbackMetric(x=X_test, y=y_test, s=S_test, verbose=self.verbose))

            # combine y and s
            y = np.concatenate([y_train, S_train], axis=1)
        else:
            # combine y and s
            y = np.concatenate([y, s], axis=1)

        self.history = super().fit(
            x,
            y,
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=self.verbose,
            validation_split=0,
            callbacks=callbacks
            ).history
        gc.collect()

        self.mean_a = np.mean(self.model_a(x)[0])
        self.std_a = np.std(self.model_a(x)[0])

        self.mean_b = np.mean(self.model_b(x[s.flatten()==self.s_max])[0])
        self.std_b = np.std(self.model_b(x[s.flatten()==self.s_max])[0])

        if self.verbose > 1:
            print("Mean / Std from Model A")
            print(self.mean_a, self.std_a)

            print("Mean / Std from Model B")
            print(self.mean_b, self.std_b)

    def predict_proba(self, x):
        return self.call_cls(x, training=False)

    def get_representations(self, x, s=None):
        if self.replace_s:
            x[s==self.s_min] = self(x[s==self.s_min], training=False)
            return x
        else:
            return self(x, training=False)

    def to_dict(self):
        """
        Return a dictionary representation of the object while dropping the tensorflow stuff.
        Useful to keep track of hyperparameters at the experiment level.
        """
        d = dict(vars(self))

        for key in dict(vars(keras.Model)).keys():
            try:
                d.pop(key)
            except KeyError:
                pass

        drop_key = []
        for key in d.keys():
            if key.startswith("_"):
                drop_key.append(key)

        for key in drop_key:
            try:
                d.pop(key)
            except KeyError:
                pass

        for key in ['optimizer', 'nan_loss', 'built', 'inputs',
                    'input_names', 'output_names', 'stop_training'
                    'history', 'train_function', 'test_function',
                    'predict_function', 'loss_tracker_a', 'loss_tracker_b',
                    'loss_tracker_cls', 'model_a', 'model_b', 'ranker',
                    'compiled_loss', 'compiled_metrics', 'loss', 'history',
                    'outputs'
                    ]:
            try:
                d.pop(key)
            except KeyError:
                pass

        return d
