# taken from https://github.com/dcmoyer/invariance-tutorial

import numpy as np
import tensorflow as tf
import keras.backend as K

import gc, math
from FairModels.BaseModel import *


#KL(N_0|N_1) = tr(\sigma_1^{-1} \sigma_0) + 
#  (\mu_1 - \mu_0)\sigma_1^{-1}(\mu_1 - \mu_0) - k +
#  \log( \frac{\det \sigma_1}{\det \sigma_0} )
def all_pairs_gaussian_kl(mu, sigma, add_third_term=False):

  sigma_sq = tf.square(sigma) + 1e-8

  #mu is [batchsize x dim_z]
  #sigma is [batchsize x dim_z]

  sigma_sq_inv = tf.math.reciprocal(sigma_sq)
  #sigma_inv is [batchsize x sizeof(latent_space)]

  #
  # first term
  #

  #dot product of all sigma_inv vectors with sigma
  #is the same as a matrix mult of diag
  first_term = tf.matmul(sigma_sq,tf.transpose(sigma_sq_inv))

  #
  # second term
  #

  #TODO: check this
  #REMEMBER THAT THIS IS SIGMA_1, not SIGMA_0

  r = tf.matmul(mu * mu,tf.transpose(sigma_sq_inv))
  #r is now [batchsize x batchsize] = sum(mu[:,i]**2 / Sigma[j])

  r2 = mu * mu * sigma_sq_inv 
  r2 = tf.reduce_sum(r2,1)
  #r2 is now [batchsize, 1] = mu[j]**2 / Sigma[j]

  #squared distance
  #(mu[i] - mu[j])\sigma_inv(mu[i] - mu[j]) = r[i] - 2*mu[i]*mu[j] + r[j]
  #uses broadcasting
  second_term = 2*tf.matmul(mu, tf.transpose(mu*sigma_sq_inv))
  second_term = r - second_term + tf.transpose(r2)

  ##uncomment to check using tf_tester
  #return second_term

  #
  # third term
  #

  # log det A = tr log A
  # log \frac{ det \Sigma_1 }{ det \Sigma_0 } =
  #   \tr\log \Sigma_1 - \tr\log \Sigma_0 
  # for each sample, we have B comparisons to B other samples...
  #   so this cancels out

  if(add_third_term):
    r = tf.reduce_sum(tf.math.log(sigma_sq),1)
    r = tf.reshape(r,[-1,1])
    third_term = r - tf.transpose(r)
  else:
    third_term = 0

  #- tf.reduce_sum(tf.log(1e-8 + tf.square(sigma)))\
  # the dim_z ** 3 term comes from
  #   -the k in the original expression
  #   -this happening k times in for each sample
  #   -this happening for k samples
  #return 0.5 * ( first_term + second_term + third_term - dim_z )
  return 0.5 * ( first_term + second_term + third_term )

#
# kl_conditional_and_marg
#   \sum_{x'} KL[ q(z|x) \| q(z|x') ] + (B-1) H[q(z|x)]
#

#def kl_conditional_and_marg(args):
def kl_conditional_and_marg(z_mean, z_log_sigma_sq, dim_z):
  z_sigma = tf.exp( 0.5 * z_log_sigma_sq )
  all_pairs_GKL = all_pairs_gaussian_kl(z_mean, z_sigma, True) - 0.5*dim_z
  return tf.reduce_mean(all_pairs_GKL)

#stolen straight from the docs
#https://keras.io/examples/variational_autoencoder/
class SamplingLayer(tf.keras.layers.Layer):
    def __init__(self, name):
        super(SamplingLayer, self).__init__(name=name)

    def call(self, x):
        z_mean, z_log_var = x
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class CvaeOutLayer(tf.keras.layers.Layer):
    def __init__(self, name, num_features, dim_z, beta_param, gamma):
        super(CvaeOutLayer, self).__init__(name=name)
        self.num_features = num_features
        self.dim_z = dim_z
        self.beta_param = beta_param
        self.gamma = gamma

    def call(self, x):
      input_x, x_hat, z_log_sigma_sq, z_mean = x
      # build reconstruction loss
      recon_loss = tf.keras.losses.mse(input_x, x_hat)
      recon_loss *= self.num_features

      # build KL loss
      kl_loss = 1 + z_log_sigma_sq - K.square(z_mean) - K.exp(z_log_sigma_sq)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      kl_qzx_qz_loss = kl_conditional_and_marg(z_mean, z_log_sigma_sq, self.dim_z)

      # optional add beta param here
      # and cite Higgins et al.
      # NOTE: we change the loss here in the paper it was:
      # (1+\lambda) * recon_loss + \beta * kl_loss + \lambda * kl_qzx_qz_loss
      loss = recon_loss + self.gamma * kl_loss + self.gamma * kl_qzx_qz_loss

      return loss


class ICVAE(BaseModel):

  def __init__(
        self,
        beta_param=0.1,
        gamma=1.0,
        dim_z=16,
        dim_s=1,
        feature_activation="tanh",
        num_hidden_layers=2,
        size_hidden_layers=20,
        batch_size=200,
        verbose=0,
        learning_rate=0.001,
        learning_rate_decay_rate=1,
        learning_rate_decay_steps=1000,
        validation_size=0.0,
        optimizer="Adam",  # 'Nadam' 'SGD'
        loss='binary_crossentropy',
        epoch=10,
        random_seed=42,
        kernel_regularizer=0.0,
        drop_out=0,
        name="ICVAE",
        run_eagerly=False
  ):

    super().__init__(
        # ICVAE HPs
        num_hidden_layers=num_hidden_layers,
        size_hidden_layers=size_hidden_layers,
        feature_activation=feature_activation,
        kernel_regularizer=kernel_regularizer, 
        drop_out=drop_out,
        # Common HPs
        loss=loss,
        validation_size=validation_size,
        batch_size=batch_size, 
        learning_rate=learning_rate,
        learning_rate_decay_rate=learning_rate_decay_rate,
        learning_rate_decay_steps=learning_rate_decay_steps,
        optimizer=optimizer,
        epoch=epoch,
        gamma=gamma,
        random_seed=random_seed,
        name=name,
        verbose=verbose,
        run_eagerly=run_eagerly
    )

    self.beta_param = beta_param
    self.dim_z = dim_z
    # they have to be the same size since they are both S in the fairness setting
    self.dim_s = dim_s

  def cvae_loss(self, y_true, y_pred):
    return K.mean(y_pred)

  def _build_model(self):

    self.x0 = tf.keras.layers.Input(
      shape=self.num_features,
      name="x0"
    )

    input_x = tf.keras.layers.Input(
        shape=self.num_features,
        name="input"
    )

    input_c = tf.keras.layers.Input(
      shape=(self.dim_s,),
      name="c"
    )

    # get encoder part
    self.enc_h = self._get_hidden_layer(
        input_layer=input_x, 
        hidden_layer=self.hidden_layers,
        drop_out=self.drop_out,
        feature_activation=self.feature_activation, 
        last_activation="", 
        reg=self.kernel_regularizer, 
        name="enc_h",
    )(self.x0)

    # get latent space part
    z_mean = tf.keras.layers.Dense(self.dim_z, activation="tanh", name="z_mean")(self.enc_h)
    z_log_sigma_sq = tf.keras.layers.Dense(self.dim_z, activation="linear", name="z_log")(self.enc_h)
    self.z = SamplingLayer(name='z')([z_mean, z_log_sigma_sq])

    # this is the concat operation!
    z_with_c = tf.keras.layers.concatenate([self.z, input_c])

    # get decoder part
    self.dec_h = self._get_hidden_layer(
        input_layer=z_with_c, 
        hidden_layer=self.hidden_layers,
        drop_out=self.drop_out, 
        feature_activation=self.feature_activation, 
        last_activation="", 
        reg=self.kernel_regularizer, 
        name="dec_h",
    )(z_with_c)

    # get x_hat
    x_hat = tf.keras.layers.Dense(self.num_features[0], name="x_hat")(self.dec_h)

    # get y_hat NOTE: binary classification only here
    y_hat = self._get_n_dense_layer(self.z, self.last_layer_size, "y_hat", "sigmoid")

    # get cvae out
    cvae_out = CvaeOutLayer(
      name = "cvae",
      num_features = self.num_features,
      dim_z = self.dim_z,
      beta_param = self.beta_param,
      gamma = self.gamma
    )([self.x0, x_hat, z_log_sigma_sq, z_mean])

    # build the model
    self.model = tf.keras.models.Model(
      inputs=[self.x0, input_c],
      outputs={
        'y_hat': y_hat,
        'cvae': cvae_out
      },
      name=self.name
    )

    if self.verbose > 1:
      self.model.summary()
      self._plot_model(self.model, "model.png")

    self.losses = {
      "y_hat": self.loss,
      "cvae": self.cvae_loss
    }

    # we set the gamma weight inside the cvae_out function
    self.lossWeights = {"y_hat": float(1 - self.gamma), "cvae": 1.0}

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      self.learning_rate,
      decay_steps=self.learning_rate_decay_steps,
      decay_rate=self.learning_rate_decay_rate,
      staircase=False
    )

    metrics = {"y_hat": "acc"}

    # compile the model
    self.model.compile(
      loss=self.losses,
      loss_weights=self.lossWeights,
      optimizer=self.optimizer(lr_schedule),
      run_eagerly=self.run_eagerly,
      metrics=metrics
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
        self.loss = 'categorical_crossentropy'
        y = self.one_hot_convert(y, self.last_layer_size)
    if len(np.unique(y)) == 2:
        y = tf.keras.utils.to_categorical(y, self.last_layer_size)

    self._build_model()

    self.history = self.model.fit(
      x=[x,s],
      y={
        'y_hat': y,
        'cvae': np.zeros(y.shape), # for the cvae we only need to model output
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
      # get the correct numpy type
      features = self.convert_array_to_float(features)

      zeros = np.zeros(features.shape[0])

      if self.last_layer_size > 1:
        res = self.model.predict([features, zeros], verbose=self.verbose)["y_hat"]
        return [0 if r[0] > r[1] else 1 for r in res]
      res = self.model.predict([features, zeros], verbose=self.verbose)["y_hat"].flatten()
      return [1 if r > threshold else 0 for r in res]

  def predict_proba(self, features):
      # get the correct numpy type
      features = self.convert_array_to_float(features)

      zeros = np.zeros(features.shape[0])
      return self.model.predict([features, zeros], verbose=self.verbose)["y_hat"][:,1]

  def get_representations(self, x):
    # get the correct numpy type
    x = self.convert_array_to_float(x)

    enc_h = self.model.get_layer('enc_h').predict(x, verbose=self.verbose)
    z_mean = self.model.get_layer('z_mean')(enc_h)
    z_log = self.model.get_layer('z_log')(enc_h)
    return self.model.get_layer('z')([z_mean, z_log]).numpy()
