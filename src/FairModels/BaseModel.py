import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from fair_metrics import group_pairwise_accuracy, auc_discrimination

from FairModels.qkerasV3 import *


colors_points = [
    (57 / 255, 106 / 255, 177 / 255),  # blue
    (218 / 255, 124 / 255, 48 / 255),  # orange
    (132 / 255, 186 / 255, 91 / 255),  # green
    (204 / 255, 37 / 255, 41 / 255),  # red
    (83 / 255, 81 / 255, 84 / 255),  # black
    (107 / 255, 76 / 255, 154 / 255),  # purple
    (171 / 255, 104 / 255, 87 / 255),  # wine
    (204 / 255, 194 / 255, 16 / 255)  # gold
]


class ReduceSum(tf.keras.layers.Layer):
    def __init__(self):
        super(ReduceSum, self).__init__()

    def call(self, x):
        return tf.reduce_sum(x, axis=1)


class EmptyLayer(tf.keras.layers.Layer):
    def __init__(self, name):
        super(EmptyLayer, self).__init__(name=name)

    def call(self, x):
        return x


class CallbackMetric(tf.keras.callbacks.Callback):
    def __init__(self, verbose, x, y, s):
        super(CallbackMetric, self).__init__()
        self.verbose = verbose
        self.x = x
        self.y = y
        self.s = s
        self._data = []

    def on_train_begin(self, logs=None):
        self._data = []

    def on_epoch_end(self, batch, logs=None):
        y_predict = np.asarray(self.model.call_cls(self.x))
        auc = roc_auc_score(self.y, y_predict)
        auc_dis = auc_discrimination(self.y, y_predict, self.s)

        self._data.append({
            'val_rocauc': auc,
            'val_auc-dis': auc_dis,
        })
        if self.verbose > 1:
            print(f"AUC {auc}, AUC-Dis {auc_dis}")
        return

    def get_data(self):
        return self._data

class TerminateOnNaN(tf.keras.callbacks.Callback):
    def __init__(self, loss_name=[], verbose=0):
        super(TerminateOnNaN, self).__init__()
        self.nan_loss = False
        self.loss_name = loss_name
        self.verbose = verbose

    def on_train_batch_end(self, batch, logs=None):
        for loss_name in self.loss_name:
            if logs[loss_name] != logs[loss_name]:
                if self.verbose > 0:
                    print("Stop bcz of nan loss in loss ", loss_name)
                self.nan_loss = True
                self.model.stop_training = True
                break

class BinaryCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 x_train=None,
                 y_train=None,
                 s_train=None,
                 size_hidden_layers=None,
                 checkpoint="",
                 plot=False,
                 loss='mi_loss',
                 gpa_gamma = 0,
                 auc_gamma = 0,
                 audc_gamma = 0,
                 multiclass=False,
                 domain_adaptation=False
                 ):
        super(BinaryCallback, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.s_train = s_train
        self.loss = loss
        self.size_hidden_layers = size_hidden_layers
        self.MIs_list = []
        self.MIy_list = []
        self.H_L_n_s0_list = []
        self.H_L_n_s1_list = []
        self.H_L_n_y0_list = []
        self.H_L_n_y1_list = []
        self.H_L_n_list = []
        self.checkpoint = checkpoint
        self.plot = plot
        self.gpa_gamma = gpa_gamma
        self.auc_gamma = auc_gamma
        self.audc_gamma = audc_gamma
        self.max_metric = 0
        self.multiclass = multiclass
        self.domain_adaptation = domain_adaptation

    def get_mi_bernoulli(self, y_true, y_pred, return_sum=True):

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

        def get_h_bernoulli(l):
            theta = np.mean(get_theta(l), axis=0)
            return -(1 - theta) * np.log2(1 - theta + 1e-20) - theta * np.log2(theta + 1e-20)

        H_L_n = get_h_bernoulli(y_pred)
        count = np.bincount(y_true)
        H_L_n_si = []
        total_si = 0
        for i in np.unique(y_true):
            si = get_h_bernoulli(y_pred[y_true == i])
            H_L_n_si.append(si)
            total_si += count[i] / len(y_true) * si
        MI = H_L_n - total_si

        if return_sum:
            return np.sum(MI), H_L_n, H_L_n_si
        else:
            return MI, H_L_n, H_L_n_si

    def get_mi(self, y_true, y_pred):

        def get_h(l):
            size = len(l)
            return -np.sum(np.unique(l, axis=0, return_counts=True)[1] / size *
                           np.log2(np.unique(l, axis=0, return_counts=True)[1] / size))

        H_L_n = get_h(y_pred)
        count = np.bincount(y_true)
        H_L_n_si = []
        total_si = 0
        for i in np.unique(y_true):
            si = get_h(y_pred[y_true == i])
            H_L_n_si.append(si)
            total_si += count[i] / len(y_true) * si
        MI = H_L_n - total_si

        return MI, H_L_n, H_L_n_si

    def getMI(self, y_pred):
        if self.loss == 'mi_loss':
            MIs, H_L_n, H_L_n_s = self.get_mi(self.s_train, y_pred)
            MIy, H_L_n, H_L_n_y = self.get_mi(self.y_train, y_pred)
            self.H_L_n_s0_list.append(H_L_n_s[0])
            self.H_L_n_s1_list.append(H_L_n_s[1])
            self.H_L_n_y0_list.append(H_L_n_y[0])
            self.H_L_n_y1_list.append(H_L_n_y[1])
        elif self.loss == 'bernoulli_loss':
            MIs, H_L_n, H_L_n_s = self.get_mi_bernoulli(self.s_train, y_pred)
            MIy, H_L_n, H_L_n_y = self.get_mi_bernoulli(self.y_train, y_pred)
            self.H_L_n_s0_list.append(H_L_n_s[0])
            self.H_L_n_s1_list.append(H_L_n_s[1])
            self.H_L_n_y0_list.append(H_L_n_y[0])
            self.H_L_n_y1_list.append(H_L_n_y[1])
        else:
            raise ValueError(f"No loss found for {self.mi_loss}")
        return MIs, MIy, H_L_n

    def on_train_batch_begin(self, batch, logs=None):
        if self.plot:

            MIs, MIy, H_L_n = self.getMI(self.model.get_layer("nn")(self.x_train))

            self.MIs_list.append(MIs)
            self.MIy_list.append(MIy)
            self.H_L_n_list.append(H_L_n)

    def on_train_end(self, logs=None):
        if self.plot:
            fig, ax_kwargs = plt.subplots(figsize=(3, 2.5))#, constrained_layout=True)

            self.model.load_weights(self.checkpoint)

            MIs, MIy, _ = self.getMI(self.model.get_layer("nn")(self.x_train))

            plt.scatter(self.MIs_list, self.MIy_list, c=np.arange(len(self.MIy_list)), marker='.', s=10)
            clb = plt.colorbar()
            clb.ax.set_title('# batch')
            plt.scatter(MIs, MIy, c=colors_points[3], marker='x', s=40)
            ax_kwargs.set_xlabel(r'$I(L_n;S)$')
            ax_kwargs.set_ylabel(r'$I(L_n;Y)$')
            #table = wandb.Table(data=[[s, y] for (s, y) in zip(self.MIs_list, self.MIy_list)],
            #                    columns=["MI_s", "MI_y"])
            #wandb.log({'MI table': table})
            #plt.savefig("results_mi/{}_mis_vs_miy.pdf".format(wandb.run.id))
            #plt.savefig("results_mi/mis_vs_miy.pdf")
            #wandb.log({"mi_tradeoff": fig})
            #wandb.log({'I(L_n;S)': self.MIs_list})
            #wandb.log({'I(L_n;Y)': self.MIy_list})
            plt.close()

            fig, ax_kwargs = plt.subplots(figsize=(3, 2.5), constrained_layout=True)
            x = [i for i in range(len(self.H_L_n_list))]
            ax_kwargs.plot(
                x,
                [np.mean(self.H_L_n_s0_list[i]) for i in range(len(self.H_L_n_list))],
                label=r'$H(L_n|S_0)$', marker='.', ms=5, color=colors_points[0], linestyle="None"
            )
            ax_kwargs.plot(
                x, [np.mean(self.H_L_n_s1_list[i]) for i in range(len(self.H_L_n_list))],
                           label=r'$H(L_n|S_1)$', marker='.', ms=5, color=colors_points[1], linestyle="None")
            ax_kwargs.plot(x, [np.mean(self.H_L_n_y0_list[i]) for i in range(len(self.H_L_n_list))],
                           label=r'$H(L_n|Y_0)$', marker='.', ms=5, color=colors_points[2], linestyle="None")
            ax_kwargs.plot(x, [np.mean(self.H_L_n_y1_list[i]) for i in range(len(self.H_L_n_list))],
                           label=r'$H(L_n|Y_1)$', marker='.', ms=5, color=colors_points[3], linestyle="None")
            plt.legend(frameon=False)
            ax_kwargs.set_xlabel('# batch')
            ax_kwargs.set_ylabel('entropy')
            plt.savefig("results_mi/entropy.pdf")
            #wandb.log({"entropy_tradeoff": fig})
            #wandb.log({"H(L_n|Y_1)": [np.mean(self.H_L_n_y1_list[i]) for i in range(len(self.H_L_n_list))]})
            #wandb.log({"H(L_n|Y_0)": [np.mean(self.H_L_n_y0_list[i]) for i in range(len(self.H_L_n_list))]})
            #wandb.log({"H(L_n|S_1)": [np.mean(self.H_L_n_s1_list[i]) for i in range(len(self.H_L_n_list))]})
            #wandb.log({"H(L_n|S_0)": [np.mean(self.H_L_n_s0_list[i]) for i in range(len(self.H_L_n_list))]})
            plt.close()

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoint != "":
            # get the predictions, since it is a sequantial model [0] is the last layer output
            pred = self.model.predict(self.x_train)[0]
            # get GPA value
            if self.gpa_gamma > 0:
                gpa = 1-group_pairwise_accuracy(self.model, self.x_train, self.y_train, self.s_train, usePred=True, pred=pred)
            if not self.multiclass:
                # get AUC value
                auc = roc_auc_score(self.y_train, pred, multi_class='ovr')
                # get audc value
                audc = 1-auc_discrimination(pred, self.y_train, self.s_train)
            else:
                pred = np.argmax(pred, axis=1)
                if self.domain_adaptation:
                    is_target_domain = np.where(self.s_train)
                    y_train_target = self.y_train[is_target_domain]
                    pred_target = pred[is_target_domain]
                    auc = accuracy_score(y_train_target, pred_target)
                else:
                    auc = accuracy_score(self.y_train, pred)
                audc = 0
                gpa = 0

            # get total metric
            metric = self.gpa_gamma * gpa + self.auc_gamma * auc + self.audc_gamma + audc
            if metric > self.max_metric:
                self.max_metric = metric
                self.model.save_weights(self.checkpoint)
            if self.verbose > 0:
                print(f"{self.gpa_gamma} x {gpa}[1-GPA] + "
                    f"{self.auc_gamma} x {auc}[AUC] + "
                    f"{self.audc_gamma} x {audc}[1-AUDC] "
                    f"cur: {metric} max: {self.max_metric}")
            return metric

class BaseModel(BaseEstimator):
    """
    TODO
    """

    def __init__(self,
        # BaseModel HPs
        hidden_layers=None,
        bias_layers=None,
        batch_normalisation_layers=None,
        quantized_position=None,
        set_quantized_position=False,
        batch_normalisation=False,
        num_hidden_layers=2,
        size_hidden_layers=20,
        num_bias_layers=2,
        size_bias_layers=20,
        feature_activation='tanh',
        activation_binary='bernoulli',
        activation_nonbinary = 'sigmoid',
        kernel_regularizer=0.0,
        drop_out=0,
        gamma=1.,
        last_layer_size=2,
        # Common HPs
        reg=0.0,
        batch_size=200,
        learning_rate=0.001,
        learning_rate_decay_rate=1,
        learning_rate_decay_steps=1000,
        optimizer="Adam",  # 'Nadam' 'SGD'
        epoch=10,
        loss='binary_crossentropy',
        ranking_loss='categorical_crossentropy',
        fair_loss='binary_crossentropy',
        # other variables
        verbose=0,
        validation_size=0.0,
        num_features=0,
        random_seed=42,
        name="BaseModel",
        dataset_name="Compas",
        interpretable=False,
        maskingLayer=None,
        nan_loss=False,
        checkpoint_path="",
        datetime="",
        run_eagerly=False,
        conv=False
    ):

        self.x1 = None
        self.x0 = None
        self.model = None
        self.maskingLayer = maskingLayer
        if maskingLayer is not None:
            self.num_dis_features = sum(self.maskingLayer)
        else:
            self.num_dis_features = 0
        self.nn_dis = None

        # DirectRanker HPs
        self.num_hidden_layers = num_hidden_layers
        self.size_hidden_layers = size_hidden_layers
        self.hidden_layers = [size_hidden_layers for i in range(num_hidden_layers)]
        if hidden_layers and num_hidden_layers == 0: self.hidden_layers = hidden_layers
        if interpretable:
            self.hidden_layers.append(num_features - self.num_dis_features)
        self.batch_normalisation_layers = batch_normalisation_layers
        self.batch_normalisation = batch_normalisation
        self.quantized_position = quantized_position
        if set_quantized_position:
            self.quantized_position = [True if i  == len(self.hidden_layers) // 2 else False for i in range(len(self.hidden_layers))]
        self.set_quantized_position = set_quantized_position
        self.num_bias_layers = num_bias_layers
        self.size_bias_layers = size_bias_layers
        self.bias_layers = [size_bias_layers for i in range(num_bias_layers)]
        if bias_layers: self.bias_layers = bias_layers
        self.feature_activation = feature_activation
        self.kernel_regularizer = kernel_regularizer
        self.activation_binary = activation_binary
        self.activation_nonbinary = activation_nonbinary
        self.drop_out = drop_out
        self.gamma = gamma
        self.last_layer_size = last_layer_size
        # Common HPs
        self.reg = reg
        self.batch_size = batch_size
        self.loss = loss
        self.ranking_loss = ranking_loss
        self.fair_loss = fair_loss
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        if optimizer == "Adam":
            self.optimizer = tf.keras.optimizers.Adam
        self.optimizer_name = optimizer
        self.epoch = epoch
        if checkpoint_path == "":
            self.checkpoint_path = checkpoint_path
        else:
            self.checkpoint_path = checkpoint_path + "/" + datetime + "/"
        self.checkpoint_path_old = checkpoint_path
        self.run_eagerly = run_eagerly
        # other variables
        self.verbose = verbose
        self.validation_size = validation_size
        self.num_features = num_features
        self.random_seed = random_seed
        self.name = name
        self.dataset_name = dataset_name
        self.interpretable = interpretable
        self.nan_loss = nan_loss
        self.conv = conv
        self.datetime = datetime

    def _plot_model(self, model, name):
        tf.keras.utils.plot_model(
            model,
            to_file=name,
            show_shapes=False,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=False,
        )

    def _get_hidden_layer(
            self,
            input_layer,
            hidden_layer=[10, 5],
            drop_out=0,
            feature_activation="tanh",
            last_activation="",
            reg=0,
            name="",
            build_bias=False,
            hybrid=False,
            bits=2,
            qactivation="bernoulli",
            qLayer=10,
            conv=False
    ):
        if not conv:
            nn = tf.keras.layers.Dense(
                units=hidden_layer[0],
                activation=feature_activation,
                kernel_regularizer=tf.keras.regularizers.l2(reg),
                bias_regularizer=tf.keras.regularizers.l2(reg),
                activity_regularizer=tf.keras.regularizers.l2(reg),
                name="nn_{}_0".format(name)
            )(input_layer)
        else:
            nn = tf.keras.layers.Conv2D(
                hidden_layer[0],
                kernel_size=(3, 3),
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(reg),
                bias_regularizer=tf.keras.regularizers.l2(reg),
                activity_regularizer=tf.keras.regularizers.l2(reg),
                name="conv_{}_0".format(name)
            )(input_layer)

        if drop_out > 0:
            nn = tf.keras.layers.Dropout(drop_out)(nn)

        for i in range(1, len(hidden_layer)):

            # if we have nominal / dis features we add a bernoulli layer for masking them
            if last_activation == "" and self.num_dis_features > 0 and i == len(hidden_layer) - 1 and self.interpretable:
                self.nn_dis = self._get_qdense(
                    nn,
                    hidden_layer=len(self.maskingLayer),
                    activation='bernoulli',
                    kernel_regularizer=tf.keras.regularizers.l2(reg),
                    bits=bits,
                    name="nn_dis"
                )

            if not conv:
                nn = tf.keras.layers.Dense(
                    units=hidden_layer[i],
                    activation=feature_activation,
                    kernel_regularizer=tf.keras.regularizers.l2(reg),
                    bias_regularizer=tf.keras.regularizers.l2(reg),
                    activity_regularizer=tf.keras.regularizers.l2(reg),
                    name="nn_{}_{}".format(name, i)
                )(nn)
            else:
                nn = tf.keras.layers.Conv2D(
                    hidden_layer[i],
                    kernel_size=(3, 3),
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(reg),
                    bias_regularizer=tf.keras.regularizers.l2(reg),
                    activity_regularizer=tf.keras.regularizers.l2(reg),
                    name="conv_{}_{}".format(name, i)
                )(nn)
                nn = tf.keras.layers.MaxPooling2D()(nn)

            if drop_out > 0 and (i < (len(hidden_layer) - 1) or last_activation != ""):
                nn = tf.keras.layers.Dropout(drop_out)(nn)

        if last_activation != "":
            nn = tf.keras.layers.Dense(
                units=1,
                activation=last_activation,
                kernel_regularizer=tf.keras.regularizers.l2(reg),
                bias_regularizer=tf.keras.regularizers.l2(reg),
                activity_regularizer=tf.keras.regularizers.l2(reg),
                name="nn_out_{}".format(name)
            )(nn)
            
        if conv:
            nn = tf.keras.layers.Flatten(name='flatten')(nn)

        if hybrid:
            nn = self._get_qdense(
                nn,
                hidden_layer=qLayer,
                activation=qactivation,
                kernel_regularizer=tf.keras.regularizers.l2(reg),
                bits=2,
                name="nn_binary"
            )
            
        if build_bias:
            hidden_part = nn
        else:
            if self.num_dis_features > 0:
                hidden_part = tf.keras.models.Model(
                    inputs=input_layer,
                    outputs=[nn, self.nn_dis],
                    name=name
                )
            else:
                hidden_part = tf.keras.models.Model(
                    inputs=input_layer,
                    outputs=nn,
                    name=name
                )

        if self.verbose == 2 and not build_bias:
            hidden_part.summary()

        return hidden_part


    def _get_quantized_activation(self, activation_str, bits=2):
        if activation_str == 'tanh':
            return 'quantized_tanh({}, symmetric=1)'.format(bits)
        elif activation_str == 'sigmoid':
            # TODO: check for symmetric=1
            return 'quantized_sigmoid({})'.format(bits)
        elif activation_str == 'det_bernoulli':
            return 'det_bernoulli'
        elif activation_str == 'bernoulli':
            return 'bernoulli'
        else:
            raise ValueError('Activation string {} not recognized'.format(activation_str))

    def _get_cls_part(
        self,
        input_layer,
        num_relevance_classes=2,
        feature_activation="softmax",
        kernel_regularizer=None,
        name="cls_part",
        index=0
    ):
        """
            Build all the output layer of the model.

            Args:
                Any: input_layer
                int: num_relevance_classes
                str: feature_activation
                Any: kernel_regularizer
                str: name
                int: index

            Returns:
                Any: output_layer
        """

        out = tf.keras.layers.Dense(
            units=num_relevance_classes,
            activation=feature_activation,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=kernel_regularizer,
            name=f"{name}_{index}"
        )(input_layer)

        return out

    def _get_qdense(
            self,
            input_layer,
            hidden_layer=10,
            activation="tanh",
            kernel_regularizer=None,
            bits=2,
            name="",
    ):
        activation = self._get_quantized_activation(activation, bits)
        nn = QDense(
            hidden_layer,
            kernel_quantizer=quantized_bits(bits),
            bias_quantizer=quantized_bits(bits),
            kernel_regularizer=kernel_regularizer,
            name=f"{name}"
        )(input_layer)
        if activation == "det_bernoulli":
            nn = deterministic_bernoulli(units=hidden_layer)(nn)
        else:
            nn = QActivation(activation, activity_regularizer=kernel_regularizer)(nn)
        return nn

    def _get_hidden_qlayer(
            self,
            input_layer,
            hidden_layer = [10, 5],
            drop_out = 0.,
            kernel_regularizer = None,
            name = "",
            conv = False,
            kernel_size = (3, 3)
    ):
        """
            Build all the hidden layers of the model.

            Args:
                Any: input_layer
                list[int]: hidden_layer
                float: drop_out
                Any: kernel_regularizer
                str: name
                conv: True if you want a convolutional net. The filter numbers will be taken from
                hidden_layer.
                kernel_size: the filter size for convnets. Ignored if conv = False.

            Returns:
                Any: output_layer
        """

        # get the activation functions for the binary part
        activation_binary = self._get_quantized_activation(self.activation_binary, 2)

        hidden_layers = input_layer
        last_quantized = None
        # loop over the number of hidden layers for the whole network
        for i in range(len(hidden_layer)):
            if self.batch_normalisation and i != 0:
                hidden_layers = tf.keras.layers.BatchNormalization()(hidden_layers)
            if conv:
                hidden_layers = tf.keras.layers.Conv2D(
                    filters=hidden_layer[i],
                    kernel_size=kernel_size,
                    activation=self.activation_nonbinary,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=kernel_regularizer,
                    name=f"{name}_{i}"
                )(hidden_layers)
            else:
                hidden_layers = tf.keras.layers.Dense(
                    units=hidden_layer[i],
                    activation=self.activation_nonbinary,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=kernel_regularizer,
                    name=f"{name}_{i}"
                )(hidden_layers)
            if self.quantized_position[i]:
                last_quantized = hidden_layers
                if activation_binary == "det_bernoulli":
                    hidden_layers = deterministic_bernoulli(units=hidden_layer[i], name=f"qact_{name}_{i}")(hidden_layers)
                else:
                    hidden_layers = QActivation(
                        activation_binary,
                        activity_regularizer=kernel_regularizer,
                        name=f"qact_{name}_{i}"
                    )(hidden_layers)
            if drop_out > 0:
                hidden_layers = tf.keras.layers.Dropout(drop_out)(hidden_layers)

        # flatten if net is convolutional
        if conv:
            hidden_layers = tf.keras.layers.Flatten()(hidden_layers)

        if self.batch_normalisation:
            hidden_layers = tf.keras.layers.BatchNormalization()(hidden_layers)

        # create classification part
        output_layer = self._get_cls_part(
            input_layer=hidden_layers,
            num_relevance_classes=self.last_layer_size,
            feature_activation=self.activation_nonbinary,
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
            name=name,
            index=len(hidden_layer)
        )

        return output_layer, last_quantized


    def _get_n_dense_layer(self, input, output_size, out_name, out_activation):

        nn = tf.keras.layers.Dense(
            units=self.hidden_layers[0],
            activation=self.feature_activation,
            kernel_regularizer=tf.keras.regularizers.l2(self.reg),
            bias_regularizer=tf.keras.regularizers.l2(self.reg),
            activity_regularizer=tf.keras.regularizers.l2(self.reg),
            name="nn_0"
        )(input)

        if self.drop_out > 0:
            nn = tf.keras.layers.Dropout(self.drop_out)(nn)

        for i in range(1, len(self.hidden_layers)):
            nn = tf.keras.layers.Dense(
                units=self.hidden_layers[i],
                activation=self.feature_activation,
                kernel_regularizer=tf.keras.regularizers.l2(self.reg),
                bias_regularizer=tf.keras.regularizers.l2(self.reg),
                activity_regularizer=tf.keras.regularizers.l2(self.reg),
                name="nn_{}".format(i)
            )(nn)

            if self.drop_out > 0:
                nn = tf.keras.layers.Dropout(self.drop_out)(nn)

        out = tf.keras.layers.Dense(
            units=output_size,
            activation=out_activation,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(self.reg),
            activity_regularizer=tf.keras.regularizers.l2(self.reg),
            name=out_name
        )(nn)

        return out


    def _get_ranking_part(
            self,
            input_layer,
            units=1,
            feature_activation="tanh",
            reg=0,
            use_bias=False,
            name="ranking_part"
    ):

        out = tf.keras.layers.Dense(
            units=units,
            activation=feature_activation,
            use_bias=use_bias,
            kernel_regularizer=tf.keras.regularizers.l2(reg),
            activity_regularizer=tf.keras.regularizers.l2(reg),
            name=name
        )(input_layer)

        return out

    def _get_ranking_qpart(
            self,
            input_layer,
            units=1,
            feature_activation='tanh',
            use_bias=False,
            kernel_regularizer=None,
            bits=2,
            name='ranking_part'
    ):

        out = QDense(
            units=units,
            use_bias=use_bias,
            kernel_quantizer=quantized_bits(bits),
            kernel_regularizer=kernel_regularizer,
            bias_quantizer=quantized_bits(bits),
            name='{}_layer'.format(name)
        )(input_layer)
        #activation = self._get_quantized_activation(feature_activation, bits)
        #out = QActivation(activation, kernel_quantizer=quantized_bits(bits),  name=name)(out)
        out = tf.keras.layers.Activation(feature_activation, name=name)(out)
        return out

    def _copy_tensor(self, t, n):
        tmp = t
        for i in range(n-1):
            tmp = tf.concat([tmp, t], 1)
        return tmp

    def _multi_tensor(self, t0, t1, n, start):
        return self._copy_tensor(t0, n) * t1[:, start:start + n]

    def _get_inter(self, nn0, nn1):
        if self.num_dis_features > 0:

            # NOTE: continues / discrete features are nn0[0] and nn1[1]

            # map the inputs to continues features
            n_num_f = self.num_features[0] - self.num_dis_features
            nn0_continues = nn0[0] + self.x0[:, :n_num_f]
            if nn1 is not None:
                nn1_continues = nn1[0] + self.x1[:, :n_num_f]

            # get correction vector
            self.correction_vector = tf.concat([nn0[0] + self.x0[:, :n_num_f], nn0[1]], 1)

            # map nn_dis to nn0/1 - one hot are right packed
            nn0_discrete = self._multi_tensor(nn0[1][:, :1], self.x0, self.maskingLayer[0], n_num_f)
            if nn1 is not None:
                nn1_discrete = self._multi_tensor(nn1[1][:, :1], self.x1, self.maskingLayer[0], n_num_f)
            n_num_f += self.maskingLayer[0]
            for idx, num in enumerate(self.maskingLayer[1:]):
                nn0_discrete = tf.concat([
                    nn0_discrete,
                    self._multi_tensor(nn0[1][:, idx+1:idx+2], self.x0, num, n_num_f)], 1)
                if nn1 is not None:
                    nn1_discrete = tf.concat([
                        nn1_discrete,
                        self._multi_tensor(nn1[1][:, idx+1:idx+2], self.x1, num, n_num_f)], 1)
                n_num_f += num

            # concate discrete and continues features
            nn0 = tf.concat([nn0_continues, nn0_discrete], 1)
            if nn1 is not None:
                nn1 = tf.concat([nn1_continues, nn1_discrete], 1)
            self.features = nn0
        else:
            if nn1 is not None:
                nn1 = nn1 + self.x1
            nn0 = nn0 + self.x0
            self.correction_vector = nn0 - self.x0
            self.features = nn0

        return nn0, nn1

    def _build_model(self):
        """
        TODO
        """
        pass

    def fit(self, x, y, s, **fit_params):
        """
        TODO
        x: numpy array of shape [num_instances, num_features]
        y: numpy array of shape [num_instances, last_layer_size]
        s: numpy array of shape [num_instances, 1]
        """
        pass
 

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

    def predict(self, features, threshold):
        """
        TODO
        """
        if len(features.shape) == 1:
            features = [features]

        res = self.model.predict([features], batch_size=self.batch_size, verbose=self.verbose)[0]

        if self.last_layer_size > 1:
            return [0 if r[0] > [1] else 1 for r in res]
        return [1 if r > threshold else 0 for r in res]

    def get_representations(self, x, split=False):
        if self.num_dis_features > 0:
            rep = self.feature_part(x).numpy()
            if split:
                return rep
            else:
                return np.concatenate([rep[0], rep[1]], 1)
        else:
            return self.feature_part(x).numpy()

    def _get_instances(self, x, y, s, samples):
        """
        :param x:
        :param y:
        :param y_bias:
        :param samples:
        """
        x0 = []
        x1 = []
        y_train = []
        s0 = []
        s1 = []

        keys, counts = np.unique(y, return_counts=True)
        sort_ids = np.argsort(keys)
        keys = keys[sort_ids]
        counts = counts[sort_ids]
        for i in range(len(keys) - 1):
            indices0 = np.random.randint(0, counts[i + 1], samples)
            indices1 = np.random.randint(0, counts[i], samples)
            querys0 = np.where(y == keys[i + 1])[0]
            querys1 = np.where(y == keys[i])[0]
            x0.extend(x[querys0][indices0])
            x1.extend(x[querys1][indices1])
            y_train.extend((keys[i + 1] - keys[i]) * np.ones(samples))
            s0.extend(s[querys0][indices0])
            s1.extend(s[querys1][indices1])

        x0 = np.array(x0)
        x1 = np.array(x1)
        s0 = np.array(s0)
        s1 = np.array(s1)
        y_train = np.array([y_train]).transpose()

        return x0, x1, y_train, s0, s1

    def train_external_rankers(self, x, y, s):
        if self.nan_loss:
            return
        lr = LogisticRegression
        rf = RandomForestClassifier

        h_train = self.get_representations(x)

        self.lr_y = lr(solver='liblinear', multi_class='ovr').fit(h_train, y)
        self.rf_y = rf(n_estimators=10).fit(h_train, y)

        self.lr_s = lr(solver='liblinear', multi_class='ovr').fit(h_train, s)
        self.rf_s = rf(n_estimators=10).fit(h_train, s)

        self.lr_y_ori = lr(solver='liblinear', multi_class='ovr').fit(x, y)
        self.rf_y_ori = rf(n_estimators=10).fit(x, y)

        self.lr_s_ori = lr(solver='liblinear', multi_class='ovr').fit(x, s)
        self.rf_s_ori = rf(n_estimators=10).fit(x, s)

    def to_dict(self):
        """
        Return a dictionary representation of the object while dropping the tensorflow stuff.
        Useful to keep track of hyperparameters at the experiment level.
        """
        d = dict(vars(self))

        for key in ['optimizer', 'nan_loss']:
            try:
                d.pop(key)
            except KeyError:
                pass

        return d

    def get_complexity(self):
        return int(np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]))

    def convert_array_to_float(self, array):
        if isinstance(array, np.ndarray):
            array = array.astype('float32')
        if isinstance(array, pd.DataFrame):
            array = array.values.astype('float32')
        return array
