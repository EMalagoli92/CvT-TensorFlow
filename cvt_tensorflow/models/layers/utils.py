import math
import warnings
from typing import Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp

from cvt_tensorflow.models.utils import _to_channel_first, _to_channel_last


def norm_cdf(x):
    """Computes standard normal cumulative distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def trunc_normal_(
    shape: Union[tf.Tensor, tuple, list],
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    dtype: Union[str, tf.dtypes.DType] = tf.float32,
) -> tf.Tensor:
    """TF2/Keras implementation of
    timm.models.layers.weight_init.trunc_normal_. Create a tf.Tensor filled
    with values drawn from a truncated normal distribution.

    Parameters
    ----------
    shape : Union[tf.Tensor, tuple, list]
        Shape of the output tensor.
    mean : float, optional
        The mean of the normal distribution.
        The default is 0.0.
    std : float, optional
        The standard deviation of the normal distribution.
        The default is 1.0.
    a : float, optional
        The minimum cutoff value.
        The default is -2.0.
    b : float, optional
        The maximum cutoff value.
        The default is 2.0.
    dtype : Union[str, tf.dtypes.DType], optional
        Dtype of the tensor.
        The default is tf.float32.

    Returns
    -------
    tf.Tensor
        Tensor filled with values drawn from a truncated normal
        distribution.
    """
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    l_ = norm_cdf((a - mean) / std)
    u_ = norm_cdf((b - mean) / std)
    tensor = tf.random.uniform(shape=shape, minval=2 * l_ - 1, maxval=2 * u_ - 1)
    tensor = tf.math.erfinv(tensor)
    tensor = tf.math.multiply(tensor, std * math.sqrt(2.0))
    tensor = tf.math.add(tensor, mean)
    tensor = tf.clip_by_value(tensor, clip_value_min=a, clip_value_max=b)
    return tf.cast(tensor, dtype)


@tf.keras.utils.register_keras_serializable(package="cvt")
class TruncNormalInitializer_(tf.keras.initializers.Initializer):
    """TF2/Keras initializer version of
    timm.models.layers.weight_init.trunc_normal_."""

    def __init__(
        self, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
    ):
        """
        Parameters
        ----------
        mean : float, optional
            The mean of the normal distribution.
            The default is 0.0.
        std : float, optional
            The standard deviation of the normal distribution.
            The default is 1.0.
        a : float, optional
            The minimum cutoff value.
            The default is -2.0.
        b : float, optional
            The maximum cutoff value.
            The default is 2.0.
        """
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

    def __call__(self, shape, dtype=None, **kwargs):
        return trunc_normal_(shape, self.mean, self.std, self.a, self.b, dtype)

    def get_config(self):
        return {"mean": self.mean, "std": self.std, "a": self.a, "b": self.b}


@tf.keras.utils.register_keras_serializable(package="cvt")
class Linear_(tf.keras.layers.Dense):
    """TF2/Keras implementation of torch.nn.Linear."""

    def __init__(
        self,
        in_features: int,
        units: int,
        use_bias: bool = True,
        kernel_initializer: Union[
            tf.keras.initializers.Initializer, str, dict
        ] = "pytorch_uniform",
        bias_initializer: Union[
            tf.keras.initializers.Initializer, str, dict
        ] = "pytorch_uniform",
        **kwargs
    ):
        """
        Parameters
        ----------
        in_features : int
            Size of each input sample.
        units : int
            Size of each output sample.
        use_bias : bool, optional
            If set to False, the layer will not learn an additive bias.
            The default is True.
        kernel_initializer : Union[tf.keras.initializers.Initializer, str, dict],
                             optional
            Initializer for the kernel weights matrix.
            If "pytorch_uniform", it will be set to Pytorch Uniform Initializer.
            The default is "pytorch_uniform".
        bias_initializer : Union[tf.keras.initializers.Initializer, str, dict], optional
            Initializer for the bias vector.
            If "pytorch_uniform", it will be set to Pytorch Uniform Initializer.
            The default is "pytorch_uniform".
        **kwargs
            Additional keyword arguments.
        """
        self.in_features = in_features

        # Initializer
        if kernel_initializer == "pytorch_uniform":
            kernel_initializer = tf.keras.initializers.RandomUniform(
                **self.uniform_initializer_spec()
            )
        if bias_initializer == "pytorch_uniform":
            bias_initializer = tf.keras.initializers.RandomUniform(
                **self.uniform_initializer_spec()
            )
        super().__init__(
            units=units,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            **kwargs
        )

    def uniform_initializer_spec(self):
        k = 1 / self.in_features
        limit = math.sqrt(k)
        return {"minval": -limit, "maxval": limit}

    def get_config(self):
        config = super().get_config()
        config.update({"in_features": self.in_features})
        return config


@tf.keras.utils.register_keras_serializable(package="cvt")
class Conv2d_(tf.keras.layers.Layer):
    """TF2/Keras implementation of torch.nn.Conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple, list],
        stride: Union[int, tuple, list] = 1,
        padding: int = 0,
        dilation: Union[int, tuple, list] = 1,
        groups: int = 1,
        bias: bool = True,
        kernel_initializer: Union[
            tf.keras.initializers.Initializer, str, dict
        ] = "pytorch_uniform",
        bias_initializer: Union[
            tf.keras.initializers.Initializer, str, dict
        ] = "pytorch_uniform",
        **kwargs
    ):
        """
        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the convolution.
        kernel_size : Union[int, tuple, list]
            Size of the convolving kernel.
        stride : Union[int, tuple, list], optional
            Stride of the convolution.
            The default is 1.
        padding : int, optional
            Padding added to all four sides of the input.
            The default is 0.
        dilation : Union[int, tuple, list], optional
            An integer or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution. Can be a single integer to
            specify the same value for all spatial dimensions.
            The default is 1.
        groups : int, optional
            A positive integer specifying the number of groups in which
            the input is split along the channel axis.
            The default is 1.
        bias : bool, optional
            If True, adds a learnable bias to the output.
            The default is True.
        kernel_initializer : Union[tf.keras.initializers.Initializer, str, dict],
                             optional
            Initializer for the kernel weights matrix.
            If "pytorch_uniform", it will be set to Pytorch Uniform Initializer.
            The default is "pytorch_uniform".
        bias_initializer : Union[tf.keras.initializers.Initializer, str, dict], optional
            Initializer for the bias vector.
            If "pytorch_uniform", it will be set to Pytorch Uniform Initializer.
            The default is "pytorch_uniform".
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        # Initializer
        if self.kernel_initializer == "pytorch_uniform":
            self.kernel_initializer = tf.keras.initializers.RandomUniform(
                **self.uniform_initializer_spec()
            )
        if self.bias_initializer == "pytorch_uniform":
            self.bias_initializer = tf.keras.initializers.RandomUniform(
                **self.uniform_initializer_spec()
            )
        self.kernel_initializer = tf.keras.initializers.get(self.kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(self.bias_initializer)

        # Data Format
        if len(tf.config.list_physical_devices("GPU")) > 0:
            self.data_format = "channels_first"
        else:
            self.data_format = "channels_last"

        # Pad Layer
        if self.padding > 0:
            self.pad_layer = tf.keras.layers.ZeroPadding2D(
                padding=padding, data_format=self.data_format
            )

        self.conv_layer = tf.keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="valid",
            data_format=self.data_format,
            dilation_rate=self.dilation,
            groups=self.groups,
            use_bias=self.bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

    def uniform_initializer_spec(self):
        # Uniform Initializer
        if isinstance(self.kernel_size, int):
            kernel_product = self.kernel_size**2
        else:
            kernel_product = self.kernel_size[0] * self.kernel_size[1]
        k = self.groups / (self.in_channels * kernel_product)
        limit = math.sqrt(k)
        return {"minval": -limit, "maxval": limit}

    def call(self, inputs, *args, **kwargs):
        if self.data_format == "channels_last":
            inputs = _to_channel_last(inputs)
        if self.padding > 0:
            x = self.pad_layer(inputs)
        else:
            x = inputs
        x = self.conv_layer(x)
        if self.data_format == "channels_last":
            x = _to_channel_first(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "dilation": self.dilation,
                "groups": self.groups,
                "bias": self.bias,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": tf.keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="cvt")
class BatchNorm2d_(tf.keras.layers.BatchNormalization):
    """TF2/Keras implementation of torch.nn.BatchNorm2d."""

    def __init__(
        self, epsilon: float = 1e-05, momentum: float = 0.9, axis: int = 1, **kwargs
    ):
        """
        Parameters
        ----------
        epsilon : float, optional
            A value added to the denominator for numerical stability.
            The default is 1e-05.
        momentum : float, optional
            The value used for the running_mean and running_var computation.
            The default is 0.9.
        axis : int, optional
            Integer, the axis that should be normalized
            (typically the features axis).
            The default is 1.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs, epsilon=epsilon, momentum=momentum, axis=axis)

    def get_config(self):
        config = super().get_config()
        return config


@tf.keras.utils.register_keras_serializable(package="cvt")
class AvgPool2d_(tf.keras.layers.Layer):
    """TF2/Keras implementation of torch.nn.AvgPool2d."""

    def __init__(
        self,
        kernel_size: Union[int, tuple],
        stride: Optional[Union[int, tuple]] = None,
        padding: Union[int, tuple] = 0,
        ceil_mode: bool = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        kernel_size : Union[int, tuple]
            The size of the window.
        stride : Optional[Union[int, tuple]], optional
            The stride of the window.
            If None, equal to kernel_size.
        padding : Union[int, tuple], optional
            Implicit zero padding to be added on both sides.
            The default is 0.
        ceil_mode : bool, optional
            When True, will use ceil instead of floor to compute the output
            shape.
            The default is False.
        **kwargs
            Additional keyword arguments.

        Note
        - kernel_size, stride, padding can be:
                - a single int – in which case the same value is used for the
                  height and width dimension.
                - a tuple of two ints – in which case, the first int is used
                  for the height dimension, and the second int for the width
                  dimension.
        - When ceil_mode=True, sliding windows are allowed to go off-bounds
          if they start within the left padding or the input.
          Sliding windows that would start in the right padded region are
          ignored.
        """
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

        # Default value for value is kernel_size
        if self.stride is None:
            self.stride = kernel_size

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)

        # Data Format
        if len(tf.config.list_physical_devices("GPU")) > 0:
            self.data_format = "channels_first"
        else:
            self.data_format = "channels_last"

        # pad Layer
        self.pad_layer = tf.keras.layers.ZeroPadding2D(
            padding=self.padding, data_format="channels_first"
        )

        # avg pool layer
        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=self.kernel_size,
            strides=self.stride,
            padding="valid",
            data_format=self.data_format,
        )

    def prepare(self, x, dtype):
        x_shape = tf.shape(x)
        H_in = tf.cast(x_shape[2], dtype)
        W_in = tf.cast(x_shape[3], dtype)

        h_out = tf.cast((H_in - self.kernel_size[0]) / self.stride[0] + 1, dtype)
        w_out = tf.cast((W_in - self.kernel_size[1]) / self.stride[1] + 1, dtype)
        # Output shape ceil mode
        H_out_ceil = tf.math.ceil(h_out)
        W_out_ceil = tf.math.ceil(w_out)

        # Output shape floor mode
        H_out_floor = tf.math.floor(h_out)
        W_out_floor = tf.math.floor(w_out)

        start_new_kernel_index = (
            H_out_floor * self.stride[0],
            W_out_floor * self.stride[1],
        )

        if not tf.equal(H_out_ceil, H_out_floor):
            n_in_rows = H_in - start_new_kernel_index[0]
            n_missing_rows = self.kernel_size[0] - n_in_rows

            if start_new_kernel_index[0] <= H_in - self.padding[0] - 1:
                new_row = tf.cast(
                    (n_missing_rows / n_in_rows), dtype
                ) * tf.math.reduce_sum(
                    input_tensor=x[
                        :, :, tf.cast(start_new_kernel_index, tf.int32)[0] :, :
                    ],
                    axis=[2],
                    keepdims=True,
                )
                zeros = tf.tile(
                    tf.zeros(tf.shape(new_row), dtype), [1, 1, n_missing_rows - 1, 1]
                )
                x = tf.concat([x, new_row, zeros], axis=2)

        if not tf.equal(W_out_ceil, W_out_floor):
            n_in_columns = W_in - start_new_kernel_index[1]
            n_missing_columns = self.kernel_size[1] - n_in_columns

            if start_new_kernel_index[1] <= W_in - self.padding[1] - 1:
                new_column = tf.cast(
                    (n_missing_columns / n_in_columns), dtype
                ) * tf.math.reduce_sum(
                    input_tensor=x[
                        :, :, :, tf.cast(start_new_kernel_index, tf.int32)[1] :
                    ],
                    axis=[3],
                    keepdims=True,
                )
                zeros = tf.tile(
                    tf.zeros(tf.shape(new_column), dtype),
                    [1, 1, 1, n_missing_columns - 1],
                )
                x = tf.concat([x, new_column, zeros], axis=3)
        return x

    def call(self, inputs, *args, **kwargs):
        x = self.pad_layer(inputs)
        if self.ceil_mode:
            x = self.prepare(x, dtype=tf.keras.backend.floatx())
        if self.data_format == "channels_last":
            x = _to_channel_last(x)
        x = self.avg_pool(x)
        if self.data_format == "channels_last":
            x = _to_channel_first(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "ceil_mode": self.ceil_mode,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="cvt")
class LayerNorm_(tf.keras.layers.LayerNormalization):
    """TF2/Keras implementation of torch.nn.LayerNorm."""

    def __init__(
        self, normalized_shape: Union[int, tuple], epsilon: float = 1e-5, **kwargs
    ):
        """
        Parameters
        ----------
        normalized_shape : Union[int, tuple]
            Input shape from an expected input of size:
            [∗ x normalized_shape[0] x normalized_shape[1] x ... x normalized_shape[−1]]
            If a single integer is used, it is treated as a singleton list,
            and this module will normalize over the last dimension which is
            expected to be of that specific size.
        epsilon : float, optional
            A value added to the denominator for numerical stability.
            The default is 1e-5.
        **kwargs
            Additional keyword arguments.
        """
        self.normalized_shape = normalized_shape
        super().__init__(**kwargs, epsilon=epsilon)

    def build(self, input_shape):

        if isinstance(self.normalized_shape, int):
            self.lnm = 1
        if isinstance(self.normalized_shape, tuple):
            self.lnm = len(self.normalized_shape)
        self.axis = tuple(range(input_shape.rank - self.lnm, input_shape.rank))
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"normalized_shape": self.normalized_shape})
        return config


@tf.keras.utils.register_keras_serializable(package="cvt")
class Identity_(tf.keras.layers.Layer):
    """TF2/Keras implementation of torch.nn.Identity."""

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args
            Any argument (unused).
        **kwargs
            Any keyword argument (unused).
        """
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        return inputs

    def get_config(self):
        config = super().get_config()
        return config


def drop_path(
    x: tf.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> tf.Tensor:
    """TF2/Keras implementation of timm.models.layers.drop_path.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor.
    drop_prob : float, optional
        Drop path probability.
        The default is 0.0.
    training : bool, optional
        Whether in training mode.
        The default is False.
    scale_by_keep : bool, optional
        To scale outputs left or not.
        The default is True.

    Returns
    -------
    tf.Tensor
        Output Tensor.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    _shape1_ = tf.expand_dims([tf.shape(x)[0]], 1)
    _shape2_ = tf.expand_dims(tf.ones((tf.rank(x) - 1), dtype=tf.int32), 0)
    shape = tf.reshape(tf.concat([_shape1_, _shape2_], 1), [-1])
    random_tensor = tfp.distributions.Bernoulli(probs=keep_prob, dtype=x.dtype).sample(
        sample_shape=shape
    )
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = tf.math.divide(random_tensor, keep_prob)
    return x * random_tensor


@tf.keras.utils.register_keras_serializable(package="cvt")
class DropPath_(tf.keras.layers.Layer):
    """TF2/Keras implementation of timm.models.layers.DropPath.

    Drop paths (Stochastic Depth) per sample (when applied in main path
    of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True, **kwargs):
        """
        Parameters
        ----------
        drop_prob : float, optional
            Drop path probability.
            The default is 0.0.
        scale_by_keep : bool, optional
            To scale outputs left or not.
            The default is True.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, inputs, training=None, *args, **kwargs):
        return drop_path(inputs, self.drop_prob, training, self.scale_by_keep)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"drop_prob": self.drop_prob, "scale_by_keep": self.scale_by_keep}
        )
        return config


@tf.keras.utils.register_keras_serializable(package="cvt")
class QuickGELU_(tf.keras.layers.Layer):
    """TF2/Keras implementation of QuickGELU present in the original CvT
    implementation."""

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        return inputs * tf.keras.layers.Activation(tf.nn.sigmoid)(1.702 * inputs)

    def get_config(self):
        config = super().get_config()
        return config
