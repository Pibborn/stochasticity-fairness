### qkeras layers which use kerasV3 ###

import warnings, logging, six

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from pyparsing import delimitedList
from pyparsing import Group
from pyparsing import Optional
from pyparsing import Regex
from pyparsing import Suppress


def _create_variable_name(attr_name, var_name=None):
  """Creates variable name.
  Arguments:
    attr_name: string. attribute name
    var_name: string. variable name

  Returns:
    string. variable name
  """

  if var_name:
    return var_name + "/" + attr_name

  # This naming scheme is to solve a problem of a layer having more than
  # one quantizer can have multiple qnoise_factor variables with the same
  # name of "qnoise_factor".
  return attr_name + "_" + str(K.get_uid(attr_name))


class BaseQuantizer(tf.Module):
  """Base quantizer

  Defines behavior all quantizers should follow.
  """

  def __init__(self):
    self.built = False

  def build(self, var_name=None, use_variables=False):
    if use_variables:
      if hasattr(self, "qnoise_factor"):
        self.qnoise_factor = tf.Variable(
            lambda: tf.constant(self.qnoise_factor, dtype=tf.float32),
            name=_create_variable_name("qnoise_factor", var_name=var_name),
            dtype=tf.float32,
            trainable=False)
      if hasattr(self, "integer"):
        self.integer = tf.Variable(
            lambda: tf.constant(self.integer, dtype=tf.int32),
            name=_create_variable_name("integer", var_name=var_name),
            dtype=tf.int32,
            trainable=False)
    self.built = True

  def _set_trainable_parameter(self):
    pass

  def update_qnoise_factor(self, qnoise_factor):
    """Update qnoise_factor."""
    if isinstance(self.qnoise_factor, tf.Variable):
      # self.qnoise_factor is a tf.Variable.
      # This is to update self.qnoise_factor during training.
      self.qnoise_factor.assign(qnoise_factor)
    else:
      if isinstance(qnoise_factor, tf.Variable):
        # self.qnoise_factor is a numpy variable, and qnoise_factor is a
        # tf.Variable.
        self.qnoise_factor = qnoise_factor.eval()
      else:
        # self.qnoise_factor and qnoise_factor are numpy variables.
        # This is to set self.qnoise_factor before building
        # (creating tf.Variable) it.
        self.qnoise_factor = qnoise_factor

  # Override not to expose the quantizer variables.
  @property
  def variables(self):
    return ()

  # Override not to expose the quantizer variables.
  @property
  def trainable_variables(self):
    return ()

  # Override not to expose the quantizer variables.
  @property
  def non_trainable_variables(self):
    return ()


def _round_through(x, use_stochastic_rounding=False, precision=0.5):
  """Rounds x but using straight through estimator.

  We use the trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182).

  Straight through estimator is a biased estimator for the rounding
  operation defined by Hinton"s Coursera Lecture 9c where dL/dx is made
  equal to dL/dy for y = f(x) during gradient computation, where f(x) is
  a non-derivable function. In that case, we assume df/dx = 1 in:

  dL   dL df   dL
  -- = -- -- = --
  dx   df dx   dy

  (https://www.youtube.com/watch?v=LN0xtUuJsEI&list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9&index=41)

  Arguments:
    x: tensor to perform round operation with straight through gradient.
    use_stochastic_rounding: if true, we perform stochastic rounding.
    precision: by default we will use 0.5 as precision, but that can overriden
      by the user.

  Returns:
    Rounded tensor.
  """
  if use_stochastic_rounding:
    output = tf_utils.smart_cond(
        K.learning_phase(),
        lambda: x + tf.stop_gradient(-x + stochastic_round(x, precision)),
        lambda: x + tf.stop_gradient(-x + tf.round(x)))
  else:
    output = x + tf.stop_gradient(-x + tf.round(x))
  return output


def _get_scaling_axis(scale_axis, len_axis):
  """Get the axis to perform auto scaling with."""

  if scale_axis is not None:
    axis = list(range(scale_axis))
    axis += list(range(scale_axis+1, len_axis))
  else:
    if K.image_data_format() == "channels_last":
      axis = list(range(len_axis - 1))
    else:
      axis = list(range(1, len_axis))
  return axis


def _get_scale(alpha, x, q, scale_axis=None, per_channel_scale=True):
  """Gets scaling factor for scaling the tensor per channel.
  It uses the least squares method to find the scaling factor.

  (https://en.wikipedia.org/wiki/Linear_least_squares)

  Arguments:
    alpha: A float or string. When it is string, it should be either "auto" or
      "auto_po2", and scale = sum(x * q, axis=all but last) / sum(q * q,
      axis=all but last)
     x: A tensor object. Its elements are in float.
     q: A tensor object. Its elements are in quantized format of x.
     scale_axis: which axis to calculate scale from
     per_channel_scale: A bool. Whether to perform per-channel scaling or not.

  Returns:
    A scaling factor tensor or scalar for scaling tensor per channel.
  """

  if isinstance(alpha, six.string_types) and "auto" in alpha:
    assert alpha in ["auto", "auto_po2"]
    # in different tensorflow version (e.g., 2.4)
    # x.shape is a tuple which doesn't have as_list() method
    try:
      x_shape = x.shape.as_list()
    except AttributeError:
      x_shape = list(x.shape)

    len_axis = len(x_shape)
    if not per_channel_scale:
      qx = K.mean(x * q, keepdims=True)
      qq = K.mean(q * q, keepdims=True)
    else:
      if len_axis > 1:
        axis = _get_scaling_axis(scale_axis, len_axis)
        qx = K.mean(tf.math.multiply(x, q), axis=axis, keepdims=True)
        qq = K.mean(tf.math.multiply(q, q), axis=axis, keepdims=True)
      else:
        # No summing (averaging) along the channel axis to get per-channel
        # scales.
        qx = x * q
        qq = q * q

    scale = qx / (qq + K.epsilon())
    if alpha == "auto_po2":
      scale = K.pow(2.0,
                    tf.math.round(K.log(scale + K.epsilon()) / np.log(2.0)))
  elif alpha is None:
    scale = 1.0
  elif isinstance(alpha, np.ndarray):
    scale = alpha
  else:
    scale = float(alpha)
  return scale



_default_sigmoid_type = "hard"
_sigmoid = None


def hard_sigmoid(x):
  """Computes hard_sigmoid function that saturates between 0 and 1."""

  return K.clip(0.5 * x + 0.5, 0.0, 1.0)


def smooth_sigmoid(x):
  """Implements a linear approximation of a sigmoid function."""

  # if we use 2.65 as the clipping point, MSE w.r.t. original sigmoid is
  # smaller than hard_simoid but the arithmetic for it is (x >> 3) +
  # (x >> 4) + 0.5, which is also not bad.

  return K.clip(0.1875 * x + 0.5, 0.0, 1.0)


def set_internal_sigmoid(mode):
  """Sets _sigmoid to either real, hard or smooth."""

  global _sigmoid

  if mode not in ["real", "hard", "smooth"]:
    raise ValueError("mode has to be 'real', 'hard' or 'smooth'.")

  if mode == "hard":
    _sigmoid = hard_sigmoid
  elif mode == "smooth":
    _sigmoid = smooth_sigmoid
  elif mode == "real":
    _sigmoid = tf.keras.backend.sigmoid


set_internal_sigmoid(_default_sigmoid_type)

class bernoulli(BaseQuantizer):  # pylint: disable=invalid-name
  """Computes a Bernoulli sample with probability sigmoid(x).

  This computation uses ST approximation.

  To do that, we compute sigmoid(x) and a random sample z ~ U[0,1]. As
  p in [0,1] and z in [0,1], p - z in [-1,1]. However, -1 will
  never appear because to get -1 we would need sigmoid(-inf) - z == 1.
  As a result, the range will be in practical terms [0,1].

  The noise introduced by z can be seen as a regularizer to the weights W of
  y = Wx as y = Wx + Wz for some noise z with mean mu(z) and var(z). As a
  result, W**2 var(z) to the variance of y, which has the same effect as a
  regularizer on L2 with lambda = var(z), as presented in Hinton"s Coursera
  Lecture 9c.

  Remember that E[dL/dy] = E[dL/dx] once we add stochastic sampling.

  Attributes:
    alpha: allows one to specify multiplicative factor for number generation
      of "auto" or "auto_po2".
    temperature: amplifier factor for sigmoid function, making stochastic
      less stochastic as it moves away from 0.
    use_real_sigmoid: use real sigmoid for probability.

  Returns:
    Computation of round with stochastic sampling with straight through
    gradient.
  """

  def __init__(self, alpha=None, temperature=6.0, use_real_sigmoid=True):
    super(bernoulli, self).__init__()
    self.alpha = alpha
    self.bits = 1
    self.temperature = temperature
    self.use_real_sigmoid = use_real_sigmoid
    self.default_alpha = 1.0
    self.scale = None

  def __str__(self):
    flags = []
    if self.alpha is not None:
      alpha = str(self.alpha)
      if isinstance(self.alpha, six.string_types):
        alpha = "'" + alpha + "'"
      flags.append("alpha=" + alpha)
    if self.temperature != 6.0:
      flags.append("temperature=" + str(self.temperature))
    if not self.use_real_sigmoid:
      flags.append("use_real_sigmoid=" + str(int(self.use_real_sigmoid)))
    return "bernoulli(" + ",".join(flags) + ")"

  def __call__(self, x):
    if isinstance(self.alpha, six.string_types):
      assert self.alpha in ["auto", "auto_po2"]

    if isinstance(self.alpha, six.string_types):
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

    if self.use_real_sigmoid:
      p = tf.keras.backend.sigmoid(self.temperature * x / std)
    else:
      p = _sigmoid(self.temperature * x/std)
    r = tf.random.uniform(tf.shape(x))
    q = tf.sign(p - r)
    q += (1.0 - tf.abs(q))
    q = (q + 1.0) / 2.0

    q_non_stochastic = tf.sign(x)
    q_non_stochastic += (1.0 - tf.abs(q_non_stochastic))
    q_non_stochastic = (q_non_stochastic + 1.0) / 2.0

    # if we use non stochastic binary to compute alpha,
    # this function seems to behave better
    scale = _get_scale(self.alpha, x, q_non_stochastic)
    self.scale = scale
    return x + tf.stop_gradient(-x + scale * q)

  def _set_trainable_parameter(self):
    if self.alpha is None:
      self.alpha = "auto_po2"

  def max(self):
    """Get the maximum value bernoulli class can represent."""
    if self.alpha is None or isinstance(self.alpha, six.string_types):
      return 1.0
    else:
      return max(1.0, self.alpha)

  def min(self):
    """Get the minimum value bernoulli class can represent."""
    return 0.0

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {"alpha": self.alpha}
    return config

class quantized_sigmoid(BaseQuantizer):  # pylint: disable=invalid-name
  """Computes a quantized sigmoid to a number of bits.

  Attributes:
    bits: number of bits to perform quantization.
    symmetric: if true, we will have the same number of values for positive
      and negative numbers.
    use_real_sigmoid: if true, will use the sigmoid from Keras backend
    use_stochastic_rounding: if true, we perform stochastic rounding.

  Returns:
    Function that performs sigmoid + quantization to bits in the range 0.0 to 1.0.
  """

  def __init__(self, bits=8, symmetric=False,
               use_real_sigmoid=False,
               use_stochastic_rounding=False):
    super(quantized_sigmoid, self).__init__()
    self.bits = bits
    self.symmetric = symmetric
    self.use_real_sigmoid = use_real_sigmoid
    self.use_stochastic_rounding = use_stochastic_rounding

  def __str__(self):
    flags = [str(self.bits)]
    if self.symmetric:
      flags.append(str(int(self.symmetric)))
    if self.use_real_sigmoid:
      flags.append(str(int(self.use_real_sigmoid)))
    if self.use_stochastic_rounding:
      flags.append(str(int(self.use_stochastic_rounding)))
    return "quantized_sigmoid(" + ",".join(flags) + ")"

  def __call__(self, x):
    x = K.cast_to_floatx(x)
    m = K.cast_to_floatx(K.pow(2, self.bits))

    p = K.sigmoid(x) if self.use_real_sigmoid else _sigmoid(x)

    return tf.keras.backend.clip((_round_through(p*m, self.use_stochastic_rounding) / m),
                                 (1.0 * self.symmetric) / m,
                                 1.0 - 1.0 / m)

  def max(self):
    """Get the maximum value that quantized_sigmoid can represent."""
    return 1.0 - 1.0 / pow(2, self.bits)

  def min(self):
    """Get the minimum value that quantized_sigmoid can represent."""
    return (1.0 * self.symmetric) / pow(2, self.bits)

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits": self.bits,
        "symmetric": self.symmetric,
        "use_real_sigmoid": self.use_real_sigmoid,
        "use_stochastic_rounding": self.use_stochastic_rounding
    }
    return config

class deterministic_bernoulli(tf.keras.layers.Layer):
    """Computes a sigmoid and samples 0 or 1 from it.

    Attributes:
        units: Integer, dimensionality of the output space.
        temperature: Float, temperature parameter for scaling the sigmoid function.
    """

    def __init__(self, units, name=None, temperature=0.6):
        """
        Initializes the layer.

        Args:
            units: Integer, dimensionality of the output space.
            name: String, name of the layer.
            temperature: Float, temperature parameter for scaling the sigmoid function.
        """
        super(deterministic_bernoulli, self).__init__(name=name)
        self.temperature = temperature

        # Initialize the threshold weight
        initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
        self.th = self.add_weight(shape=(units, ), initializer=initializer, trainable=True)

    def __str__(self):
        """
        Returns a string representation of the layer.
        """
        return "deterministic_bernoulli"

    def call(self, x):
        """
        Forward pass of the layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with sigmoid quantization.
        """
        # Compute probabilities using sigmoid function
        std = 1.0
        p = tf.keras.backend.sigmoid(self.temperature * x / std)

        # Compute binary quantization based on threshold
        q = tf.cast(tf.math.greater(p, self.th), tf.float32)

        # Compute non-binary quantization for potential use
        q_non_binary = tf.sign(x)
        q_non_binary += (1.0 - tf.abs(q_non_binary))
        q_non_binary = (q_non_binary + 1.0) / 2.0

        # Combine input and quantized output
        return x * self.th + tf.stop_gradient(-x * self.th + q)


class QActivation(tf.keras.layers.Layer):
  """Implements quantized activation layers."""

  def __init__(self, activation, **kwargs):

    super(QActivation, self).__init__(**kwargs)

    self.activation = activation

    if not isinstance(activation, six.string_types):
      self.quantizer = activation
      if hasattr(self.quantizer, "__name__"):
        self.__name__ = self.quantizer.__name__
      elif hasattr(self.quantizer, "name"):
        self.__name__ = self.quantizer.name
      elif hasattr(self.quantizer, "__class__"):
        self.__name__ = self.quantizer.__class__.__name__
      return

    self.__name__ = activation

    try:
      self.quantizer = get_quantizer(activation)
    except KeyError:
      raise ValueError("invalid activation '{}'".format(activation))

  def call(self, inputs):
    return self.quantizer(inputs)

  def get_config(self):
    config = {"activation": self.activation}
    base_config = super(QActivation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return str(self.activation)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_prunable_weights(self):
    return []


class quantized_bits(BaseQuantizer):  # pylint: disable=invalid-name
  """Quantizes the number to a number of bits.

  In general, we want to use a quantization function like:

  a = (pow(2,bits) - 1 - 0) / (max(x) - min(x))
  b = -min(x) * a

  in the equation:

  xq = a x + b

  This requires multiplication, which is undesirable. So, we
  enforce weights to be between -1 and 1 (max(x) = 1 and min(x) = -1),
  and separating the sign from the rest of the number as we make this function
  symmetric, thus resulting in the following approximation.

  1) max(x) = +1, min(x) = -1
  2) max(x) = -min(x)

  a = pow(2,bits-1)
  b = 0

  Finally, just remember that to represent the number with sign, the
  largest representation is -pow(2,bits) to pow(2, bits-1)

  Symmetric and keep_negative allow us to generate numbers that are symmetric
  (same number of negative and positive representations), and numbers that
  are positive.

  Note:
    the behavior of quantized_bits is different than Catapult HLS ac_fixed
    or Vivado HLS ap_fixed. For ac_fixed<word_length, integer_lenth, signed>,
    when signed = true, it is equavlent to
    quantized_bits(word_length, integer_length-1, keep_negative=True)

  Attributes:
    bits: number of bits to perform quantization.
    integer: number of bits to the left of the decimal point.
    symmetric: if true, we will have the same number of values for positive
      and negative numbers.
    alpha: a tensor or None, the scaling factor per channel.
      If None, the scaling factor is 1 for all channels.
    keep_negative: if true, we do not clip negative numbers.
    use_stochastic_rounding: if true, we perform stochastic rounding.
    scale_axis: which axis to calculate scale from
    qnoise_factor: float. a scalar from 0 to 1 that represents the level of
      quantization noise to add. This controls the amount of the quantization
      noise to add to the outputs by changing the weighted sum of
      (1 - qnoise_factor)*unquantized_x + qnoise_factor*quantized_x.
    var_name: String or None. A variable name shared between the tf.Variables
      created in the build function. If None, it is generated automatically.
    use_ste: Bool. Whether to use "straight-through estimator" (STE) method or
        not.
    use_variables: Bool. Whether to make the quantizer variables to be dynamic
      tf.Variables or not.

  Returns:
    Function that computes fixed-point quantization with bits.
  """

  def __init__(self,
               bits=8,
               integer=0,
               symmetric=0,
               keep_negative=True,
               alpha=None,
               use_stochastic_rounding=False,
               scale_axis=None,
               qnoise_factor=1.0,
               var_name=None,
               use_ste=True,
               use_variables=False):
    super(quantized_bits, self).__init__()
    self.bits = bits
    self.integer = integer
    self.symmetric = symmetric
    self.keep_negative = keep_negative
    self.alpha = alpha
    self.use_stochastic_rounding = use_stochastic_rounding
    # "auto*" |-> symmetric
    if isinstance(self.alpha, six.string_types):
      self.symmetric = True
    self.scale = None
    self.scale_axis = scale_axis
    self.qnoise_factor = qnoise_factor
    self.use_ste = use_ste
    self.var_name = var_name
    self.use_variables = use_variables

  def __str__(self):
    # Convert Tensors to printable strings by converting to a numpy array and
    # then using regex to remove brackets when there is only one integer bit
    integer_bits = re.sub(
        r"\[(\d)\]", r"\g<1>",
        str(self.integer.numpy() if isinstance(self.integer, tf.Variable
                                              ) else self.integer))

    flags = [str(self.bits), integer_bits, str(int(self.symmetric))]
    if not self.keep_negative:
      flags.append("keep_negative=False")
    if self.alpha:
      alpha = str(self.alpha)
      if isinstance(self.alpha, six.string_types):
        alpha = "'" + alpha + "'"
      flags.append("alpha=" + alpha)
    if self.use_stochastic_rounding:
      flags.append("use_stochastic_rounding=" +
                   str(int(self.use_stochastic_rounding)))
    return "quantized_bits(" + ",".join(flags) + ")"

  def __call__(self, x):
    """Computes fixedpoint quantization of x."""
    if not self.built:
      self.build(var_name=self.var_name, use_variables=self.use_variables)

    x = K.cast_to_floatx(tf.convert_to_tensor(x))

    # quantized_bits with "1" bit becomes a binary implementation.
    unsigned_bits = self.bits - self.keep_negative
    m = K.cast_to_floatx(pow(2, unsigned_bits))
    m_i = K.cast_to_floatx(K.pow(2, self.integer))

    if self.alpha is None:
      scale = 1.0
    elif isinstance(self.alpha, six.string_types):
      # We only deal with the symmetric case right now.
      assert self.symmetric, "Only symmetric quantizers are implemented"
      len_axis = len(x.shape)
      if len_axis > 1:
        axis = _get_scaling_axis(self.scale_axis, len_axis)
      else:
        axis = [0]

      x = x / m_i

      # Using 2's complement, we can represent 2**(bits-1)-1 positive values
      # If we wish to maintain symmetry, we can double 2**(bits-1)-1 to get
      # the total number of possible values we can represent.
      # If symmetry is not enforced, then we can represent (2**bits)-1 values
      # using 2's complement.
      levels = (2**(self.bits-1)-1) * 2 if self.symmetric else (2**self.bits)-1

      scale = (K.max(abs(x), axis=axis, keepdims=True) * 2) / levels

      # If alpha is "auto_po2", then get the "best" po2 scale
      if "po2" in self.alpha:
        scale = K.pow(2.0,
                      tf.math.round(K.log(scale + K.epsilon()) / np.log(2.0)))
        for _ in range(5):
          v = tf.floor(tf.abs(x) / scale + 0.5)
          mask = v < levels / 2
          z = tf.sign(x) * tf.where(mask, v, tf.ones_like(v) * levels / 2)
          scale = _get_scale(alpha="auto_po2", x=x, q=z,
                             scale_axis=self.scale_axis)

      # If alpha is "auto", then get the "best" floating point scale
      elif self.alpha == "auto":
        v = tf.floor(tf.abs(x) / scale + 0.5)
        mask = v < levels / 2
        z = tf.sign(x) * tf.where(mask, v, tf.ones_like(v) * levels / 2)
      else:
        raise ValueError(f"Invalid alpha '{self.alpha}'")

      # z is an integer number, so we must make the scale * m and z / m
      scale = scale * m

      # we will not use "z" right now because of stochastic_rounding
      # this is still under test.

      # if "new" in self.alpha:
      #  z = z / m
      #  self.scale = scale
      #  return x + tf.stop_gradient(-x + scale * z)
      x = m_i * x
      xq = m_i * z / m
      self.scale = scale
      xq = scale * xq

      if self.use_ste:
        return x + tf.stop_gradient(self.qnoise_factor * (-x + xq))
      else:
        return (1 - self.qnoise_factor) * x + tf.stop_gradient(
            self.qnoise_factor * xq)

    else:
      scale = self.alpha

    # quantized_bits with "1" bit becomes a binary implementation.
    if unsigned_bits > 0:
      p = x * m / m_i
      xq = m_i * K.clip(
          _round_through(p, self.use_stochastic_rounding, precision=1.0),
          self.keep_negative  * (-m + self.symmetric), m - 1) / m
    else:
      xq = tf.sign(x)
      xq += (1.0 - tf.abs(xq))
      if not self.keep_negative:
        xq = (xq + 1.0) / 2.0

    self.scale = scale
    xq = scale * xq

    if self.use_ste:
      return x + tf.stop_gradient(self.qnoise_factor * (-x + xq))
    else:
      return (1 - self.qnoise_factor) * x + tf.stop_gradient(
          self.qnoise_factor * xq)

  def _set_trainable_parameter(self):
    if self.alpha is None:
      self.alpha = "auto_po2"
      self.symmetric = True

  def max(self):
    """Get maximum value that quantized_bits class can represent."""
    unsigned_bits = self.bits - self.keep_negative
    if unsigned_bits > 0:
      return max(
          1.0,
          np.array(
              K.pow(2., K.cast(self.integer, dtype="float32")),
              dtype="float32"))
    else:
      return 1.0

  def min(self):
    """Get minimum value that quantized_bits class can represent."""
    if not self.keep_negative:
      return 0.0
    unsigned_bits = self.bits - self.keep_negative
    if unsigned_bits > 0:
      return -max(
          1.0,
          np.array(
              tf.keras.ops.pow(2, tf.keras.ops.cast(self.integer, dtype="float32")), dtype="float32"))
    else:
      return -1.0

  def range(self):
    """Returns a list of all values that quantized_bits can represent
    ordered by their binary representation ascending."""
    assert self.symmetric == 0
    assert self.keep_negative
    assert self.alpha is None or self.alpha == 1.0

    x = np.asarray(range(2**self.bits), dtype=np.float32)
    p_and_n = np.where(x >= 2**(self.bits - 1),
                       (x - 2**(self.bits - 1)) - 2**(self.bits - 1), x)
    return p_and_n * np.array(
        tf.keras.ops.pow(2.0, -self.bits + tf.keras.ops.cast(self.integer, dtype="float32") + 1),
        dtype="float32")

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    config = {
        "bits":
            self.bits,
        "integer":
            self.integer.numpy()
            if isinstance(self.integer, tf.Variable) else self.integer,
        "symmetric":
            self.symmetric,
        "alpha":
            self.alpha,
        "keep_negative":
            self.keep_negative,
        "use_stochastic_rounding":
            self.use_stochastic_rounding,
        "qnoise_factor":
            self.qnoise_factor.numpy() if isinstance(
                self.qnoise_factor, tf.Variable) else self.qnoise_factor
    }
    return config


class Clip(tf.keras.constraints.Constraint):
  """Clips weight constraint."""

  # This function was modified from Keras minmaxconstraints.
  #
  # Constrains the weights to be between min/max values.
  #   min_value: the minimum norm for the incoming weights.
  #   max_value: the maximum norm for the incoming weights.
  #   constraint: previous constraint to be clipped.
  #   quantizer: quantizer to be applied to constraint.

  def __init__(self, min_value=0.0, max_value=1.0,
               constraint=None, quantizer=None):
    """Initializes Clip constraint class."""

    self.min_value = min_value
    self.max_value = max_value
    self.constraint = tf.keras.constraints.get(constraint)
    # Don't wrap yourself
    if isinstance(self.constraint, Clip):
      self.constraint = None
    self.quantizer = get_quantizer(quantizer)

  def __call__(self, w):
    """Clips values between min and max values."""
    if self.constraint:
      w = self.constraint(w)
      if self.quantizer:
        w = self.quantizer(w)
    w = K.clip(w, self.min_value, self.max_value)
    return w

  def get_config(self):
    """Returns configuration of constraint class."""
    return {"min_value": self.min_value, "max_value": self.max_value}

  @classmethod
  def from_config(cls, config):
    if isinstance(config.get('constraint', None), Clip):
      config['constraint'] = None
    config['constraint'] = tf.keras.constraints.get(config.get('constraint', None))
    config['quantizer'] = get_quantizer(config.get('quantizer', None))
    return cls(**config)


def get_initializer(identifier):
  """Gets the initializer.

  Args:
    identifier: An initializer, which could be dict, string, or callable function.

  Returns:
    A initializer class

  Raises:
    ValueError: An error occurred when quantizer cannot be interpreted.
  """
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    if identifier['class_name'] == 'QInitializer':
      return QInitializer.from_config(identifier['config'])
    else:
      return tf.keras.initializers.get(identifier)
  elif isinstance(identifier, six.string_types):
    return tf.keras.initializers.get(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError("Could not interpret initializer identifier: " +
                     str(identifier))

class QInitializer(tf.keras.initializers.Initializer):
  """Wraps around Keras initializer to provide a fanin scaling factor."""

  def __init__(self, initializer, use_scale, quantizer):
    self.initializer = initializer
    self.use_scale = use_scale
    self.quantizer = quantizer

    try:
      self.is_po2 = "po2" in quantizer.__class__.__name__
    except:
      self.is_po2 = False

  def __call__(self, shape, dtype=None):
    x = self.initializer(shape, dtype)

    max_x = np.max(abs(x))
    std_x = np.std(x)
    delta = self.quantizer.max() * 2**-self.quantizer.bits

    # delta is the minimum resolution of the number system.
    # we want to make sure we have enough values.
    if delta > std_x and hasattr(self.initializer, "scale"):
      q = self.quantizer(x)
      max_q = np.max(abs(q))
      scale = 1.0
      if max_q == 0.0:
        xx = np.mean(x * x)
        scale = self.quantizer.max() / np.sqrt(xx)
      else:
        qx = np.sum(q * x)
        qq = np.sum(q * q)

        scale = qq / qx

      self.initializer.scale *= max(scale, 1)
      x = self.initializer(shape, dtype)

    return np.clip(x, -self.quantizer.max(), self.quantizer.max())

  def get_config(self):
    return {
        "initializer": self.initializer,
        "use_scale": self.use_scale,
        "quantizer": self.quantizer,
    }

  @classmethod
  def from_config(cls, config):
    config = {
      'initializer' : get_initializer(config['initializer']),
      'use_scale'   : config['use_scale'],
      'quantizer'   : get_quantizer(config['quantizer'])}
    return cls(**config)


def get_constraint(identifier, quantizer):
  """Gets the initializer.

  Args:
    identifier: A constraint, which could be dict, string, or callable function.
    quantizer: A quantizer class or quantization function

  Returns:
    A constraint class
  """
  if identifier:
    if isinstance(identifier, dict) and identifier['class_name'] == 'Clip':
      return Clip.from_config(identifier['config'])
    else:
      return tf.keras.constraints.get(identifier)
  else:
    max_value = max(1, quantizer.max()) if hasattr(quantizer, "max") else 1.0
    return Clip(-max_value, max_value, identifier, quantizer)


def get_auto_range_constraint_initializer(quantizer, constraint, initializer):
  """Get value range automatically for quantizer.

  Arguments:
   quantizer: A quantizer class in quantizers.py.
   constraint: A tf.keras constraint.
   initializer: A tf.keras initializer.

  Returns:
    a tuple (constraint, initializer), where
      constraint is clipped by Clip class in this file, based on the
      value range of quantizer.
      initializer is initializer contraint by value range of quantizer.
  """
  if quantizer is not None:
    constraint = get_constraint(constraint, quantizer)
    initializer = get_initializer(initializer)

    if initializer and initializer.__class__.__name__ not in ["Ones", "Zeros", 'QInitializer']:
      # we want to get the max value of the quantizer that depends
      # on the distribution and scale
      if not (hasattr(quantizer, "alpha") and
              isinstance(quantizer.alpha, six.string_types)):
        initializer = QInitializer(
            initializer, use_scale=True, quantizer=quantizer)
  return constraint, initializer


def Num(s):
  """Tries to convert string to either int or float."""
  try:
    try:
      return int(s)
    except ValueError:
      return float(s)
  except ValueError:
    # this should be always true. if it isn't int or float, it should be str
    assert (
        (s[0] == '"' and s[-1] == '"') or
        (s[0] == "'" and s[-1] == "'")
    )
    s = s[1:-1]
    return s

def Str(s):
  return s[1:-1]

def IsNum(s):
  try:
    try:
      int(s)
      return True
    except ValueError:
      float(s)
      return True
  except ValueError:
    return False

def IsBool(s):
  if s in ["True", "False"]:
    return True
  else:
    return False

def Bool(s):
  return True if "True" in s else False

def GetArg(s):
  if IsBool(s):
    return Bool(s)
  elif IsNum(s):
    return Num(s)
  else:
    return Str(s)


def GetParams(s):
  """Extracts args and kwargs from string."""
  # modified from https://stackoverflow.com/questions/38799223/parse-string-to-identify-kwargs-and-args  # pylint: disable=line-too-long

  _lparen = Suppress("(")  # pylint: disable=invalid-name
  _rparen = Suppress(")")  # pylint: disable=invalid-name
  _eq = Suppress("=")  # pylint: disable=invalid-name

  data = (_lparen + Optional(
      delimitedList(
          Group(Regex(r"[^=,)\s]+") + Optional(_eq + Regex(u"[^,)]*")))
          )
      ) + _rparen)

  items = data.parseString(s).asList()

  # need to make sure that kwargs only happen after args are processed
  args = [GetArg(i[0]) for i in items if len(i) == 1]
  kwargs = {i[0]: GetArg(i[1]) for i in items if len(i) == 2}

  # check for syntax error
  for i in range(1, len(items)):
    if (len(items[i]) == 1) and (len(items[i-1]) == 2):
      raise SyntaxError

  return args, kwargs


def safe_eval(eval_str, op_dict, *params, **kwparams):  # pylint: disable=invalid-name
  """Replaces eval by a safe eval mechanism."""

  function_split = eval_str.split("(")
  quantizer = op_dict.get(function_split[0], None)

  if len(function_split) == 2:
    args, kwargs = GetParams("(" + function_split[1])
  else:
    args = []
    kwargs = {}

  args = args + list(params)
  for k in kwparams:
    kwargs[k] = kwparams[k]

  # must be Keras activation object if None
  if quantizer is None:
    logging.info("keras dict %s", function_split[0])
    quantizer = tf.keras.activations.get(function_split[0])

  if len(function_split) == 2 or args or kwargs:
    return quantizer(*args, **kwargs)
  else:
    if isinstance(quantizer, type):
      # Check if quantizer is a class
      return quantizer()
    else:
      # Otherwise it is a function, so just return it
      return quantizer


def get_quantizer(identifier):
  """Gets the quantizer.

  Args:
    identifier: An quantizer, which could be dict, string, or callable function.

  Returns:
    A quantizer class or quantization function from this file. For example,
      Quantizer classes: quantized_bits, quantized_po2, quantized_relu_po2,
      binary, stochastic_binary, ternary, stochastic_ternary, etc.

      Quantization functions: binary_sigmoid, hard_sigmoid, soft_sigmoid, etc.

  Raises:
    ValueError: An error occurred when quantizer cannot be interpreted.
  """

  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return tf.keras.utils.deserialize_keras_object(
        identifier, module_objects=globals(), printable_module_name="quantizer")
  elif isinstance(identifier, six.string_types):
    return safe_eval(identifier, globals())
  elif callable(identifier):
    return identifier
  else:
    raise ValueError("Could not interpret quantizer identifier: " +
                     str(identifier))


class QDense(tf.keras.layers.Dense):
  """Implements a quantized Dense layer."""

  # Most of these parameters follow the implementation of Dense in
  # Keras, with the exception of kernel_range, bias_range,
  # kernel_quantizer, bias_quantizer, and kernel_initializer.
  #
  # kernel_quantizer: quantizer function/class for kernel
  # bias_quantizer: quantizer function/class for bias
  # kernel_range/bias_ranger: for quantizer functions whose values
  #   can go over [-1,+1], these values are used to set the clipping
  #   value of kernels and biases, respectively, instead of using the
  #   constraints specified by the user.
  #
  # we refer the reader to the documentation of Dense in Keras for the
  # other parameters.

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer="he_normal",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_quantizer=None,
               bias_quantizer=None,
               kernel_range=None,
               bias_range=None,
               **kwargs):

    if kernel_range is not None:
      warnings.warn("kernel_range is deprecated in QDense layer.")

    if bias_range is not None:
      warnings.warn("bias_range is deprecated in QDense layer.")

    self.kernel_range = kernel_range
    self.bias_range = bias_range

    self.kernel_quantizer = kernel_quantizer
    self.bias_quantizer = bias_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    # optimize parameter set to "auto" scaling mode if possible
    if hasattr(self.kernel_quantizer_internal, "_set_trainable_parameter"):
      self.kernel_quantizer_internal._set_trainable_parameter()

    self.quantizers = [
        self.kernel_quantizer_internal, self.bias_quantizer_internal
    ]

    kernel_constraint, kernel_initializer = (
        get_auto_range_constraint_initializer(self.kernel_quantizer_internal,
                                              kernel_constraint,
                                              kernel_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))
    if activation is not None:
      activation = get_quantizer(activation)

    super(QDense, self).__init__(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

  def call(self, inputs):
    if self.kernel_quantizer:
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel
    output = K.dot(inputs, quantized_kernel)
    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias
      output = K.bias_add(output, quantized_bias,
                                         data_format="channels_last")
    if self.activation is not None:
      output = self.activation(output)
    return output

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) >= 2
    assert input_shape[-1]
    output_shape = list(input_shape)
    output_shape[-1] = self.units
    return tuple(output_shape)

  def get_config(self):
    config = {
        "units": self.units,
        "activation": activations.serialize(self.activation),
        "use_bias": self.use_bias,
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "kernel_initializer":
            initializers.serialize(self.kernel_initializer),
        "bias_initializer":
            initializers.serialize(self.bias_initializer),
        "kernel_regularizer":
            regularizers.serialize(self.kernel_regularizer),
        "bias_regularizer":
            regularizers.serialize(self.bias_regularizer),
        "activity_regularizer":
            regularizers.serialize(self.activity_regularizer),
        "kernel_constraint":
            constraints.serialize(self.kernel_constraint),
        "bias_constraint":
            constraints.serialize(self.bias_constraint),
        "kernel_range": self.kernel_range,
        "bias_range": self.bias_range
    }
    base_config = super(QDense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return {
        "kernel_quantizer":
            str(self.kernel_quantizer_internal),
        "bias_quantizer":
            str(self.bias_quantizer_internal),
        "activation":
            str(self.activation),
        "units" : str(self.units)
    }

  def get_quantizers(self):
    return self.quantizers

  def get_prunable_weights(self):
    return [self.kernel]
