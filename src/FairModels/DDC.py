from FairModels.BaseModel import *
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from FairModels.mmd import maximum_mean_discrepancy


def domain_loss(y_true, y_pred):
    mask = tf.reshape(tf.cast(y_true, tf.bool), (tf.size(y_true),))
    Xs = tf.boolean_mask(y_pred, mask)
    Xt = tf.boolean_mask(y_pred, ~mask)
    return maximum_mean_discrepancy(Xs, Xt)

class DeepDomainConfusion(BaseModel):
    """
    TODO
    """
    def __init__(self,
            # DDC HPs
            num_hidden_layers_1=2,
            num_hidden_layers_2=2,
            size_hidden_layers_1=20,
            size_hidden_layers_2=20,
            feature_activation='tanh',
            kernel_regularizer=0.0,
            drop_out=0,
            gamma=0.5,
            # Common HPs
            batch_size=200,
            learning_rate=0.001,
            learning_rate_decay_rate=1,
            learning_rate_decay_steps=1000,
            optimizer="Adam",  # 'Nadam' 'SGD'
            epoch=10,
            # other variables
            verbose=0,
            validation_size=0.0,
            random_seed=42,
            name="DeepDomainConfusion",
            dataset_name="Compas",
            interpretable=False,
            nan_loss=False,
            checkpoint_path="",
            datetime="",
            run_eagerly=False
            ):

        super().__init__(
            # DirectRankerAdv HPs
            feature_activation=feature_activation,
            kernel_regularizer=kernel_regularizer,
            drop_out=drop_out,
            gamma=gamma,
            # Common HPs
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_decay_rate=learning_rate_decay_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            optimizer=optimizer,
            epoch=epoch,
            # other variables
            verbose=verbose,
            validation_size=validation_size,
            random_seed=random_seed,
            name=name,
            dataset_name=dataset_name,
            interpretable=interpretable,
            nan_loss=nan_loss,
            checkpoint_path=checkpoint_path,
            datetime=datetime,
            run_eagerly=run_eagerly
        )

        self.num_hidden_layers_1 = num_hidden_layers_1 
        self.size_hidden_layers_1 = size_hidden_layers_1
        self.num_hidden_layers_2 = num_hidden_layers_2 
        self.size_hidden_layers_2 = size_hidden_layers_2
        self.hidden_layers_1 = [size_hidden_layers_1 for i in range(num_hidden_layers_1)]
        self.hidden_layers_2 = [size_hidden_layers_2 for i in range(num_hidden_layers_2)]

    def _build_model(self):
        # Placeholders for the inputs

        input_layer = tf.keras.layers.Input(
            shape=self.num_features,
            name="input"
        )

        input_layer_2 = tf.keras.layers.Input(
            shape=(int(self.size_hidden_layers_1), ),
            name="input_2"
        )

        self.nn = self._get_hidden_layer(
            input_layer,
            hidden_layer=self.hidden_layers_1,
            drop_out=self.drop_out,
            feature_activation=self.feature_activation,
            reg=self.kernel_regularizer,
            name='nn',
        )

        last_acti_out = self.nn(input_layer)

        nn_2 = self._get_hidden_layer(
            input_layer_2,
            hidden_layer=self.hidden_layers_2,
            drop_out=self.drop_out,
            feature_activation=self.feature_activation,
            reg=self.kernel_regularizer,
            name='nn_2',
        )

        out = tf.keras.layers.Dense(
            self.last_layer_size,
            activation="sigmoid",
            name="classification"
        )(nn_2(last_acti_out))

        self.model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=[out, last_acti_out],
            name=self.name
        )

        if self.verbose > 1:
            self.model.summary()

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=self.learning_rate_decay_steps,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False
        )

        losses = {
            "classification": tf.keras.losses.BinaryCrossentropy(from_logits=True),
            "nn": domain_loss
        }
        lossWeights = {
            "classification": float(1.0 - self.gamma),
            "nn": float(self.gamma)
        }
        metrics = {"classification": "acc"}

        self.model.compile(
            optimizer=self.optimizer(lr_schedule),
            loss=losses,
            loss_weights=lossWeights,
            metrics=metrics,
            run_eagerly=self.run_eagerly
        )

    def fit(self, x, y, s, **fit_params):
        """
        TODO
        x: numpy array of shape [num_instances, num_features]
        y: numpy array of shape [num_instances, last_layer_size]
        s: numpy array of shape [num_instances, 1]
        """
        # set seed
        tf.random.set_seed(self.random_seed)

        # get the correct numpy type
        x = self.convert_array_to_float(x)
        y = self.convert_array_to_float(y)
        s = self.convert_array_to_float(s)

        self.num_features = (x.shape[1], )

        # convert for cls but not for external rankers
        self.last_layer_size = len(np.unique(y))
        if len(np.unique(y)) != 2:
            self.ranking_loss = 'categorical_crossentropy'
            y = self.one_hot_convert(y, self.last_layer_size)
        if len(np.unique(y)) == 2:
            y = tf.keras.utils.to_categorical(y, self.last_layer_size)

        self._build_model()

        CallbackOnNaN = TerminateOnNaN()

        ydict = {"classification": y, "nn": s}

        self.history = self.model.fit(
            x=x,
            y=ydict,
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=self.verbose,
            shuffle=True,
            validation_split=0,
            callbacks=[CallbackOnNaN]
        ).history

    def predict_proba(self, features):
        """
        TODO
        """
        features = self.convert_array_to_float(features)
        if len(features.shape) == 1:
            features = [features]

        res = self.model.predict([features], batch_size=self.batch_size, verbose=self.verbose)[0]

        if self.last_layer_size > 1:
            return res[:,1]
        return res

    def predict(self, features, threshold=0.5):
        """
        TODO
        """
        features = self.convert_array_to_float(features)
        if len(features.shape) == 1:
            features = [features]

        res = self.model.predict([features], batch_size=self.batch_size, verbose=self.verbose)[0]

        res = keras.activations.sigmoid(res)

        if self.last_layer_size > 1:
            return [0 if r[0] > [1] else 1 for r in res]
        return [1 if r > threshold else 0 for r in res]

    def get_representations(self, x, split=False):
        x = self.convert_array_to_float(x)
        return self.nn.predict(x)
