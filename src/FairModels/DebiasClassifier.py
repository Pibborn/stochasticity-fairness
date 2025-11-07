import numpy as np
import tensorflow as tf

import gc
from FairModels.BaseModel import *
from FairModels.GradReverse import GradReverse


class DebiasClassifier(BaseModel):
    """
    TODO
    """

    def __init__(self,
            # DebiasClassifier HPs
            hidden_layers=None,
            bias_layers=None,
            num_hidden_layers=2,
            size_hidden_layers=20,
            num_bias_layers=2,
            size_bias_layers=20,
            feature_activation='tanh',
            kernel_regularizer=0.0,
            drop_out=0,
            drop_out_bias=0,
            gamma=1.,
            # Common HPs
            batch_size=200,
            learning_rate=0.001,
            learning_rate_decay_rate=1,
            learning_rate_decay_steps=1000,
            optimizer="Adam",# 'Nadam' 'SGD'
            epoch=10,
            ranking_loss='binary_crossentropy',
            fair_loss='binary_crossentropy',
            # other variables
            verbose=0,
            validation_size=0.0,
            random_seed=42,
            name="DebiasClassifier",
            dataset_name="Compas",
            interpretable=False,
            maskingLayer=None,
            nan_loss=False,
            datetime="",
            run_eagerly=False,
            conv=False 
            ):
    
        super().__init__(
            # DebiasClassifier HPs
            hidden_layers=hidden_layers,
            bias_layers=bias_layers,
            num_hidden_layers=num_hidden_layers,
            size_hidden_layers=size_hidden_layers,
            num_bias_layers=num_bias_layers,
            size_bias_layers=size_bias_layers,
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
            ranking_loss=ranking_loss, 
            fair_loss=fair_loss,
            # other variables
            verbose=verbose,
            validation_size=validation_size,
            random_seed=random_seed, 
            name=name,
            dataset_name=dataset_name,
            interpretable=interpretable,
            maskingLayer=maskingLayer,
            nan_loss=nan_loss,
            datetime=datetime,
            run_eagerly=run_eagerly,
            conv=conv
            )

        # DebiasClassifier HPs
        self.drop_out_bias = drop_out_bias

    def _build_model(self):
        """
        TODO
        """

        # Placeholders for the inputs
        self.x0 = tf.keras.layers.Input(
            shape=self.num_features,
            name="x0"
        )

        input_layer = tf.keras.layers.Input(
            shape=self.num_features,
            name="input"
        )

        self.feature_part = self._get_hidden_layer(
            input_layer=input_layer, 
            hidden_layer=self.hidden_layers,
            drop_out=self.drop_out, 
            feature_activation=self.feature_activation, 
            last_activation="", 
            reg=self.kernel_regularizer, 
            name="nn",
        )

        if self.verbose > 1:
            self.feature_part.summary()
            self._plot_model(self.feature_part, "feature_part.png")

        # Cls Part
        nn0 = self.feature_part(self.x0)

        if self.interpretable:
            nn0, _ = self._get_inter(nn0, None)
        
        # Debias Layers
        nn_bias_0 = GradReverse()(nn0)

        nn_bias_0 = self._get_hidden_layer(
            input_layer=nn_bias_0, 
            hidden_layer=self.bias_layers, 
            drop_out=self.drop_out_bias, 
            feature_activation=self.feature_activation, 
            last_activation="sigmoid", 
            reg=self.kernel_regularizer, 
            name="bias",
            build_bias=True
        )

        out = self._get_ranking_part(
            input_layer=nn0,
            units=self.last_layer_size,
            feature_activation="sigmoid",
            reg=self.kernel_regularizer,
            use_bias=True,
            name="cls_part"
        )

        self.model = tf.keras.models.Model(
            inputs=self.x0,
            outputs=[out, nn_bias_0],
            name=self.name
        )

        if self.verbose > 1:
            self.model.summary()
            self._plot_model(self.model, "model.png")

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=self.learning_rate_decay_steps,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False
        )

        losses = {
            "cls_part" : self.ranking_loss,
            "nn_out_bias" : self.fair_loss,
        }

        lossWeights = {"cls_part": float(1 - self.gamma), "nn_out_bias": float(self.gamma)}
        metrics = {'cls_part': 'AUC' if self.last_layer_size == 1 else 'acc', 'nn_out_bias': 'acc'}

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

        ydict = {"cls_part": y, "nn_out_bias": s}

        self.history = self.model.fit(
            x=x,
            y=ydict,
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=self.verbose,
            shuffle=True,
            validation_split=self.validation_size,
            callbacks=[CallbackOnNaN]
        ).history
        # https://github.com/tensorflow/tensorflow/issues/14181
        # https://github.com/tensorflow/tensorflow/issues/30324
        gc.collect()

        self.nan_loss = CallbackOnNaN.nan_loss

    def one_hot_convert(self, y, num_classes):
        arr = np.zeros((len(y), num_classes))
        for i, yi in enumerate(y):
            arr[i, int(yi)-1] = 1
        return arr
