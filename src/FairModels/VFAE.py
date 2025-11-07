import numpy as np
import tensorflow as tf

import gc
from FairModels.BaseModel import *


class VFAE(BaseModel):
    """
    TODO
    """

    def __init__(self,
            # VFAE HPs
            num_hidden_layers=2,
            size_hidden_layers=20,
            dim_z=20,
            beta_param=1.,
            alpha_param=1.,
            D=500,
            feature_activation='tanh',
            kernel_regularizer=0.0,
            drop_out=0,
            gamma=1.,
            # Common HPs
            batch_size=200,
            learning_rate=0.001,
            learning_rate_decay_rate=1,
            learning_rate_decay_steps=1000,
            optimizer="Adam",# 'Nadam' 'SGD'
            epoch=10,
            loss='binary_crossentropy',
            # other variables
            verbose=0,
            validation_size=0.0,
            random_seed=42,
            name="VFAE",
            dataset_name="Compas",
            run_eagerly=False,
            conv=False
        ):

        super().__init__(
            # DebiasClassifier HPs
            num_hidden_layers=num_hidden_layers,
            size_hidden_layers=size_hidden_layers,
            feature_activation=feature_activation,
            kernel_regularizer=kernel_regularizer, 
            drop_out=drop_out,
            gamma=gamma,
            # Common HPs
            loss=loss,
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
            run_eagerly=run_eagerly,
            conv=conv
            )

        self.D = D
        self.beta_param = beta_param
        self.alpha_param = alpha_param
        self.dim_z = dim_z

    def KL(self, mu1, log_sigma_sq1, mu2=0., log_sigma_sq2=0.):
        return 0.5*ReduceSum()(log_sigma_sq2-log_sigma_sq1-1+(tf.keras.ops.exp(log_sigma_sq1)+tf.keras.ops.power(mu1-mu2,2))/tf.keras.ops.exp(log_sigma_sq2))

    def fast_MMD(self, x1, x2):
        inner_difference = tf.reduce_mean(self.psi(x1), axis=0) - tf.reduce_mean(self.psi(x2), axis=0)
        return tf.tensordot(inner_difference, inner_difference, axes=1)

    def psi(self, x):
        W = tf.random.normal([self.dim_z, self.D])
        b = tf.random.uniform([self.D], 0, 2 * np.pi)

        # NOTE: we use gamma here as defined in https://arxiv.org/pdf/1511.00830.pdf E.q. 10 and D=500
        # FastMMD(z) = \sqrt(\frac{2/D}) \cdot cos(\frac{2/\gamma} W z + b)
        # for the value we use 2xM of the dim_z as defined in https://arxiv.org/pdf/1806.09918.pdf E.q. 33
        gamma = 2 * self.dim_z
        return tf.sqrt(2 / self.D) * tf.cos(tf.cast(tf.sqrt(2 / gamma), tf.float32) * tf.matmul(x, W) + b)

    def KL_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_pred)

    def vfae_loss(self, y_true, y_pred):
        # MMD loss
        input_s = y_true
        z1_enc = y_pred
        MMD_x1 = tf.boolean_mask(z1_enc, tf.tile(tf.cast(input_s, tf.bool), [1, tf.shape(z1_enc)[1]]))
        MMD_x2 = tf.boolean_mask(z1_enc,tf.tile(tf.cast(1-input_s, tf.bool),[1, tf.shape(z1_enc)[1]]))
        MMD = self.fast_MMD(tf.reshape(MMD_x1,[tf.cast(tf.shape(MMD_x1)[0]/tf.shape(z1_enc)[1],tf.int32),tf.shape(z1_enc)[1]]),
                    tf.reshape(MMD_x2,[tf.cast(tf.shape(MMD_x2)[0]/tf.shape(z1_enc)[1],tf.int32),tf.shape(z1_enc)[1]]))
        return MMD

    def _build_model(self):
        """
        TODO
        """

        # Placeholders for the inputs
        self.x0 = tf.keras.layers.Input(
            shape=self.num_features,
            name="x0"
        )

        input_x = tf.keras.layers.Input(
            shape=self.num_features,
            name="inputKL_x"
        )

        input_s = tf.keras.layers.Input(
            shape=self.num_s,
            name="input_s"
        )

        input_y = tf.keras.layers.Input(
            shape=(self.last_layer_size, ),
            name="input_y"
        )

        # get encoder part
        self.enc_h = self._get_hidden_layer(
            input_layer=input_x,
            hidden_layer=self.hidden_layers,
            drop_out=self.drop_out,
            feature_activation=self.feature_activation, 
            last_activation="", 
            reg=self.kernel_regularizer, 
            name="enc_h"
        )(self.x0)

        # get latent space part
        def get_z_part(input, name):
            epsilon = tf.random.normal([self.dim_z], dtype=tf.float32, name='epsilon0')
            hidden = tf.keras.layers.Dense(self.dim_z, activation=self.feature_activation, name="z_hidden_"+name)(input)
            mean = tf.keras.layers.Dense(self.dim_z, activation="linear", name="z_mean_"+name)(hidden)
            sigma = tf.keras.layers.Dense(self.dim_z, activation="linear", name="z_log_"+name)(hidden)
            return mean + tf.keras.ops.exp(sigma / 2) * epsilon, mean, sigma

        # encoder latent space
        self.z1_enc, z1_enc_mu, z1_enc_log_sigma = get_z_part(
            tf.keras.layers.Concatenate(axis=1)([self.enc_h, input_s]),
            "z1_enc"
        )
        self.z1_enc = EmptyLayer(name="MMD")(self.z1_enc)
        self.z2_enc, z2_enc_mu, z2_enc_log_sigma = get_z_part(
            tf.keras.layers.Concatenate(axis=1)([self.z1_enc, input_y]),
            "z2_enc"
        )

        # decoder latent space
        self.z1_dec, z1_dec_mu, z1_dec_log_sigma = get_z_part(
            tf.keras.layers.Concatenate(axis=1)([self.z2_enc, input_y]),
            "z1_dec"
        )
        self.z2_dec = tf.keras.layers.Concatenate(axis=1)([self.z1_dec, input_s])

        # get dencoder part
        self.dec_h = self._get_hidden_layer(
            input_layer=self.z2_dec,
            hidden_layer=self.hidden_layers,
            drop_out=self.drop_out,
            feature_activation=self.feature_activation, 
            last_activation="", 
            reg=self.kernel_regularizer, 
            name="dec_h"
        )(self.z2_dec)

        # get x_hat
        x_hat = tf.keras.layers.Dense(self.num_features[0]+self.num_s[0], activation="linear", name="x_hat")(self.dec_h)

        # get y_hat NOTE: binary classification only here
        y_hat = self._get_n_dense_layer(self.z1_enc, self.last_layer_size, "y_hat", "sigmoid")

        # get KL loss function
        KL_z1 = self.KL(z1_enc_mu, z1_enc_log_sigma, z1_dec_mu, z1_dec_log_sigma)
        KL_z1 = EmptyLayer(name="KL_z1")(KL_z1)
        KL_z2 = self.KL(z2_enc_mu, z2_enc_log_sigma)
        KL_z2 = EmptyLayer(name="KL_z2")(KL_z2)

        # build the model
        self.model = tf.keras.models.Model(
            inputs=[self.x0, input_s, input_y],
            outputs={
                "MMD": self.z1_enc,
                'x_hat': x_hat,
                'y_hat': y_hat,
                'KL_z1': KL_z1,
                'KL_z2': KL_z2
            },
            name=self.name
        )

        if self.verbose > 1:
            self.model.summary()
            self._plot_model(self.model, "model.png")

        self.losses = {
            "MMD": self.vfae_loss,
            "x_hat": tf.keras.losses.MeanSquaredError(),
            "y_hat": self.loss,
            'KL_z1': self.KL_loss,
            'KL_z2': self.KL_loss
        }

        # NOTE: the paper takes different values for \alpha adult, Health and German -> \alpha=1
        # Amazon reviews and Extended Yale B \alpha = 100 * (N_batch_source + N_batch_target) / N_batch_source
        self.lossWeights = {"MMD": float(self.gamma), "x_hat": float(self.alpha_param), "y_hat": float(1 - self.gamma), "KL_z1": float(self.gamma), "KL_z2": float(self.gamma)}

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=self.learning_rate_decay_steps,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False
        )

        # compile the model
        self.model.compile(
            optimizer=self.optimizer(lr_schedule),
            loss=self.losses,
            loss_weights=self.lossWeights,
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
        # NOTE: we only go with binary S and y
        if len(np.unique(y)) > 2:
            raise ValueError("ERROR we only support binary classes at the moment.")

        # TODO: this is hardcoded and will fail in the future
        self.num_s = (1, )

        if len(s.shape) == 1:
            s = s.reshape([len(s), 1])

        # convert for cls but not for external rankers
        self.last_layer_size = len(np.unique(y))
        if len(np.unique(y)) != 2:
            self.loss = 'categorical_crossentropy'
            y = self.one_hot_convert(y, self.last_layer_size)
        if len(np.unique(y)) == 2:
            y = tf.keras.utils.to_categorical(y, self.last_layer_size)

        self._build_model()

        # fill up batch size NOTE: MMD loss can have NaN values so batch_size should be not to small
        if len(x) % self.batch_size != 0:
            x = np.concatenate([x, x[:self.batch_size-len(x) % self.batch_size]])
            y = np.concatenate([y, y[:self.batch_size-len(y) % self.batch_size]])
            s = np.concatenate([s, s[:self.batch_size-len(s) % self.batch_size]])

        self.history = self.model.fit(
            x=[x,s,y],
            y={
                "MMD": s,
                'x_hat': np.append(x, s, axis=1),
                'y_hat': y,
                'KL_z1': np.zeros(y.shape), # for the KL we only need to model output
                'KL_z2': np.zeros(y.shape)
            },
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=self.verbose,
            shuffle=True,
            validation_split=self.validation_size
        ).history
        # https://github.com/tensorflow/tensorflow/issues/14181
        # https://github.com/tensorflow/tensorflow/issues/30324
        gc.collect()

    def predict(self, features, threshold=0.5):
        zeros_s = np.zeros(features.shape[0])
        zeros_y = zeros_s
        if self.last_layer_size > 1:
            zeros_y = np.zeros((features.shape[0], 2))

        if self.last_layer_size > 1:
            res = self.model.predict([features, zeros_s, zeros_y], verbose=self.verbose)["y_hat"]
            return [0 if r[0] > r[1] else 1 for r in res]
        res = self.model.predict([features, zeros_s, zeros_y], verbose=self.verbose)["y_hat"].flatten()
        return [1 if r > threshold else 0 for r in res]
    
    def print_model_outputs(self, output_dict):
        print('x_hat: ', output_dict['x_hat'])
        print('y_hat: ', output_dict['y_hat'])

    def predict_proba(self, features, debug=False):
        features = self.convert_array_to_float(features)
        zeros_s = np.zeros(features.shape[0])
        zeros_y = zeros_s
        if self.last_layer_size > 1:
            zeros_y = np.zeros((features.shape[0], 2))
            pred_dict = self.model.predict([features, zeros_s, zeros_y], verbose=self.verbose)
            if debug:
                self.print_model_outputs(pred_dict)
            return pred_dict["y_hat"][:,1]
        else:
            pred_dict = self.model.predict([features, zeros_s, zeros_y], verbose=self.verbose)
            if debug:
                self.print_model_outputs(pred_dict)
            return pred_dict["y_hat"]

    def get_representations(self, x):
        x = self.convert_array_to_float(x)
        epsilon = tf.random.normal([self.dim_z], dtype=tf.float32)
        enc_h = self.model.get_layer('enc_h').predict(x, verbose=self.verbose)
        z_hidden = self.model.get_layer('z_hidden_z1_enc')(np.append(enc_h, np.zeros([enc_h.shape[0], 1]), axis=1))
        z_mean = self.model.get_layer('z_mean_z1_enc')(z_hidden)
        z_log = self.model.get_layer('z_log_z1_enc')(z_hidden)
        rep = z_mean + tf.exp(z_log / 2) * epsilon
        return rep.numpy()
