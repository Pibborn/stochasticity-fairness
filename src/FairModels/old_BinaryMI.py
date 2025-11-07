from FairModels.old_BaseModel import *
import tensorflow as tf
from tensorflow.python.ops import gen_array_ops
from FairModels.qkerasV3 import *
import gc
import six
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import pandas as pd


class DomainAdaptationModel(tf.keras.Model):

    def __init__(self, binary_model):
        self.binary_model = binary_model

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        y_true = y['cls_part']
        s_true = y['nn']

        with tf.GradientTape() as tape:
            y_pred = self.binary_model(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class old_BinaryMI(BaseModel):
    """
    TODO
    """

    def __init__(self,
                 # BinaryMI HPs
                 num_hidden_layers=2,
                 size_hidden_layers=20,
                 feature_activation='tanh',
                 last_feature_activation='bernoulli',
                 kernel_regularizer=0.01,
                 drop_out=0,
                 gamma=1.,
                 gamma_y=1.,
                 # Common HPs
                 batch_size=200,
                 learning_rate=0.001,
                 learning_rate_decay_rate=1,
                 learning_rate_decay_steps=1000,
                 optimizer="Adam",  # 'Nadam' 'SGD'
                 epoch=10,
                 cls_loss='binary_crossentropy',
                 mi_loss='mi_loss',  # or 'bernoulli_loss'
                 run_eagerly=False,
                 # other variables
                 verbose=0,
                 validation_size=0.1,
                 random_seed=42,
                 name="old_BinaryMI",
                 dataset_name="Compas",
                 interpretable=False,
                 nan_loss=False,
                 bits=2,
                 checkpoint_path="",
                 datetime="",
                 plot=False,
                 hybrid=False,
                 hybrid_layer=20,
                 conv=False,
                 gpa_gamma=1,
                 auc_gamma=1,
                 audc_gamma=1,
                 last_layer_size=1,
                 domain_adaptation=False,
                 num_sensitive_values=2,
                 callbacks=False
                 ):
        super().__init__(
            # DirectRankerAdv HPs
            num_hidden_layers=num_hidden_layers,
            size_hidden_layers=size_hidden_layers,
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

        self.cls_loss = cls_loss
        self.last_feature_activation = last_feature_activation
        self.bits = bits
        self.gamma_y = gamma_y
        self.mi_loss = mi_loss
        self.plot = plot
        self.hybrid = hybrid
        self.hybrid_layer = hybrid_layer
        self.conv = conv
        self.gpa_gamma = gpa_gamma
        self.auc_gamma = auc_gamma
        self.audc_gamma = audc_gamma
        self.last_layer_size = last_layer_size
        self.domain_adaptation = domain_adaptation
        self.num_sensitive_values = num_sensitive_values
        self.callbacks = callbacks

    def custom_classification_loss(self, y_true, y_pred):
        s_true = y_true[:, 1]
        y_true = y_true[:, 0]
        return tf.math.multiply(self.classification_loss(y_true, y_pred), tf.cast(1 - s_true, tf.float32))

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
        num_classes = self.num_sensitive_values
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
            cnt_i = tf.shape(y_i)[0]  # number of repr with index i
            norm_si = cnt_i / tf.shape(y_pred)[0]
            norm_s.append(norm_si)

        norm_s = tf.convert_to_tensor(norm_s)
        H_L_n_s = tf.convert_to_tensor(H_L_n_s)
        MI = H_L_n - tf.reduce_sum(tf.math.multiply(norm_s, H_L_n_s))
        return MI

    def max_mutual_information_loss(self, y_true, y_pred):
        return -self.mutual_information_loss(y_true, y_pred)

    def mutual_information_loss(self, y_true, y_pred):
        """
        I((x1,...,xn);y)= H((x1,...,xn))   - H((x1,...,xn)|y)
                        = H((x1,...,xn)) - H((x1,...,xn)|s)
                        = H((x1,...,xn)) - (H((x1,...,xn)|s=0) + H((x1,...,xn)|s=1))

        pseudocode:
        def get_h(l):
            size = len(l)
            return -np.sum(np.unique(l, axis=0, return_counts=True)[1]/size *
                           np.log2(np.unique(l, axis=0, return_counts=True)[1]/size))
        ps = 0.6
        num = 2000
        y_pred = np.random.binomial(n=1, p=ps, size=[num, 5])
        y_true = np.random.binomial(n=1, p=ps, size=[num])

        y_pred[y_true == 0] = np.random.binomial(n=1, p=p0, size=[len(y_true[y_true == 0]), 5])
        y_pred[y_true == 1] = np.random.binomial(n=1, p=p1, size=[len(y_true[y_true == 1]), 5])

        H_L_n = get_h(y_pred)
        H_L_n_s0 = get_h(y_pred[y_true == 0])
        H_L_n_s1 = get_h(y_pred[y_true == 1])

        counts = np.bincount(y_true)

        H_L_n_list.append(H_L_n)
        H_L_n_s0_list.append(counts[0] / num * H_L_n_s0)
        H_L_n_s1_list.append(counts[1] / num * H_L_n_s1)

        MI = H_L_n - ((counts[0] / num * H_L_n_s0) + (counts[1] / num * H_L_n_s1))

        :param y_pred: output of the layer
        :param y_true: sensitive attribute
        :return: The loss
        """

        def log2(x):
            numerator = tf.math.log(x + 1e-20)
            denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
            return numerator / denominator

        def get_h(tensor):
            size = tf.shape(tensor)[0]
            # https://stackoverflow.com/questions/57861344/how-to-apply-unique-with-counts-over-2d-array-in-tensorflow#57862096
            y, idx, count = gen_array_ops.unique_with_counts_v2(tensor, [0])
            return -tf.reduce_sum(count / size * log2(count / size))  # tf.reduce_sum(count/size)

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float64)
        num_classes = self.num_sensitive_values
        H_L_n = get_h(y_pred)
        H_L_n_s = []
        norm_s = []
        for i in range(num_classes):
            if len(y_true.shape) == 1:
                y_filter = tf.where(y_true == i)
            else:
                y_filter = tf.where(y_true[:, 0] == i)[:, 0]
            y_i = tf.gather(y_pred, indices=y_filter)
            H_L_n_si = get_h(y_i)
            H_L_n_s.append(H_L_n_si)
            cnt_i = tf.shape(y_i)[0]  # number of repr with index i
            norm_si = cnt_i / tf.shape(y_pred)[0]
            norm_s.append(norm_si)

        norm_s = tf.convert_to_tensor(norm_s)
        H_L_n_s = tf.convert_to_tensor(H_L_n_s)

        MI = H_L_n - tf.reduce_sum(tf.math.multiply(norm_s, H_L_n_s))

        return MI

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

        if self.hybrid:
            self.feature_part = self._get_hidden_layer(
                input_layer,
                hidden_layer=self.hidden_layers,
                drop_out=self.drop_out,
                feature_activation=self.feature_activation,
                last_activation="",
                reg=self.kernel_regularizer,
                name='nn',
                hybrid=self.hybrid,
                bits=self.bits,
                qactivation=self.last_feature_activation,
                qLayer=self.hybrid_layer,
                conv=self.conv
            )
        else:
            self.feature_part = self._get_hidden_qlayer(
                input_layer,
                hidden_layer=self.hidden_layers,
                drop_out=self.drop_out,
                feature_activation=self.feature_activation,
                last_activation=self.last_feature_activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
                name='nn',
                bits=self.bits,
                conv=self.conv
            )

        # Ranking Part
        if self.mi_loss == 'bernoulli_loss':
            nn_last = self.feature_part.layers[1](input_layer)
            for layer in self.feature_part.layers[2:-1]:
                nn_last = layer(nn_last)
            nn_last = tf.keras.models.Model(input_layer, nn_last, name="bernoulli_input")
            last_acti_out = nn_last(self.x0)
        else:
            last_acti_out = self.feature_part(self.x0)
        nn0 = self.feature_part(self.x0)

        if self.interpretable:
            nn0 = nn0 + self.x0
            self.correction_vector = nn0 - self.x0
            self.features = nn0

        if self.hybrid:
            out = self._get_ranking_part(
                input_layer=nn0,
                units=self.last_layer_size,
                feature_activation="sigmoid" if self.last_layer_size == 1 else "softmax",
                reg=self.kernel_regularizer,
                use_bias=True,
                name="cls_part"
            )
        else:
            out = self._get_ranking_qpart(
                input_layer=nn0,
                units=self.last_layer_size,
                feature_activation="sigmoid" if self.last_layer_size == 1 else "softmax",
                kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
                use_bias=True,
                name="cls_part"
            )

        self.model = tf.keras.models.Model(
            inputs=self.x0,
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

        if self.cls_loss == 'binary_crossentropy':
            self.classification_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        elif self.cls_loss == 'sparse_categorical_crossentropy':
            self.classification_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        if self.mi_loss == 'mi_loss':
            # "nn_1": self.max_mutual_information_loss
            losses = {
                "cls_part": self.custom_classification_loss if self.domain_adaptation else self.cls_loss,
                "nn": self.mutual_information_loss
            }
            # "nn_1": self.gamma_y
            lossWeights = {"cls_part": 1 - self.gamma, "nn": self.gamma}
            # 'nn_1': 'acc'
            metrics = {'cls_part': 'AUC' if self.last_layer_size == 1 else 'acc', 'nn': 'acc'}

        elif self.mi_loss == 'bernoulli_loss':
            # "nn_1": self.max_mutual_information_loss
            losses = {
                "cls_part": self.custom_classification_loss if self.domain_adaptation else self.cls_loss,
                "bernoulli_input": self.mutual_information_bernoulli_loss
            }
            # "nn_1": self.gamma_y
            lossWeights = {"cls_part": float(1 - self.gamma), "bernoulli_input": float(self.gamma)}
            # 'nn_1': 'acc'
            metrics = {'cls_part': 'AUC' if self.last_layer_size == 1 else 'acc', 'bernoulli_input': 'acc'}
        else:
            raise ValueError(f"No loss found for {self.mi_loss}")

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
        """
        # set seed
        tf.random.set_seed(self.random_seed)

        # get the correct numpy type
        x = self.convert_array_to_float(x)
        y = self.convert_array_to_float(y)
        s = self.convert_array_to_float(s)

        self.num_features = (x.shape[1], )
        self._build_model()

        if self.checkpoint_path != "" and self.plot:
            plot = True
        else:
            plot = False

        if self.callbacks:
            x_train, x_vali, y_train, y_vali, s_train, s_vali = train_test_split(x, y, s,
                                                                                test_size=self.validation_size,
                                                                                random_state=self.random_seed)

            CallbackBinary = BinaryCallback(
                x_train=x_vali,
                y_train=y_vali,
                s_train=s_vali,
                size_hidden_layers=self.size_hidden_layers,
                checkpoint=self.checkpoint_path,
                plot=self.plot,
                loss=self.mi_loss,
                gpa_gamma=self.gpa_gamma,
                auc_gamma=self.auc_gamma,
                audc_gamma=self.audc_gamma,
                multiclass=self.last_layer_size > 1,
                domain_adaptation=self.domain_adaptation
            )
            callbacks = [CallbackBinary]

        if not self.callbacks:
            x_train = x
            y_train = y
            s_train = s
            callbacks = None

        if self.mi_loss == 'mi_loss':
            # "nn_1": y_train
            ydict = {"cls_part": np.array((y_train, s_train)).T if self.domain_adaptation else y_train,
                     "nn": s_train}
        elif self.mi_loss == 'bernoulli_loss':
            # "nn_1": y_train
            ydict = {"cls_part": np.array((y_train, s_train)).T if self.domain_adaptation else y_train,
                     "bernoulli_input": s_train}
        else:
            raise ValueError(f"No loss found for {self.mi_loss}")

        self.history = self.model.fit(
            x=x_train,
            y=ydict,
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=self.verbose,
            shuffle=True,
            validation_split=0,
            callbacks=callbacks
        ).history
        # https://github.com/tensorflow/tensorflow/issues/14181
        # https://github.com/tensorflow/tensorflow/issues/30324
        gc.collect()

        if self.checkpoint_path != "":
            self.model.load_weights(self.checkpoint_path)

    def predict_proba(self, features):
        """
        TODO
        """
        # get correct type
        if isinstance(features, np.ndarray):
            features = features.astype('float32')
        if isinstance(features, pd.DataFrame):
            features = features.values.astype('float32')

        if len(features.shape) == 1:
            features = [features]

        res = self.model.predict(features, batch_size=self.batch_size, verbose=self.verbose)[0]

        return res
