from typing import Literal, Optional

import tensorflow as tf
import tensorflow_addons as tfa

from cvt_tensorflow.models.layers.utils import (
    Linear_,
    QuickGELU_,
    TruncNormalInitializer_,
)


@tf.keras.utils.register_keras_serializable(package="cvt")
class Mlp(tf.keras.layers.Layer):
    """Multi-Layer Perceptron (MLP) block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: str = "gelu",
        drop: float = 0.0,
        init: Literal["trunc_norm", "xavier"] = "trunc_norm",
        **kwargs
    ):
        """
        Parameters
        ----------
        in_features : int
            Input features dimension.
        hidden_features : Optional[int], optional
            Hidden features dimension.
            The default is None.
        out_features : Optional[int], optional
            Output features dimension.
            The default is None.
        act_layer : str, optional
            Name of activation layer.
            The default is "gelu".
        drop : float, optional
            Dropout rate.
            The default is 0.0.
        init : Literal["trunc_norm", "xavier"], optional
            Initialization method.
            Possible values are: "trunc_norm", "xavier".
            The default is "trunc_norm".
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        self.drop = drop
        self.init = init

    def build(self, input_shape):
        self._out_features = self.out_features or self.in_features
        self._hidden_features = self.hidden_features or self.in_features
        if self.init == "xavier":
            init_ = tf.keras.initializers.GlorotUniform()
        elif self.init == "trunc_norm":
            init_ = TruncNormalInitializer_(std=0.02)
        else:
            raise ValueError(
                "Unknown initialization method({}). "
                "Possible values are: 'trunc_norm', 'xavier'.".format(self.init)
            )
        self.fc1 = Linear_(
            in_features=self.in_features,
            units=self._hidden_features,
            kernel_initializer=init_,
            bias_initializer=tf.keras.initializers.Zeros(),
            name="fc1",
        )
        if self.act_layer == "gelu":
            self.act = tfa.layers.GELU(approximate=False, name="act")
        elif self.act_layer == "quick_gelu":
            self.act = QuickGELU_(name="act")
        else:
            self.act = tf.keras.layers.Activation(
                self.act_layer, dtype=self.dtype, name="act"
            )
        self.fc2 = Linear_(
            in_features=self._hidden_features,
            units=self._out_features,
            kernel_initializer=init_,
            bias_initializer=tf.keras.initializers.Zeros(),
            name="fc2",
        )
        self._drop = tf.keras.layers.Dropout(rate=self.drop, name="drop")
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self._drop(x)
        x = self.fc2(x)
        x = self._drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
                "act_layer": self.act_layer,
                "drop": self.drop,
                "init": self.init,
            }
        )
        return config
