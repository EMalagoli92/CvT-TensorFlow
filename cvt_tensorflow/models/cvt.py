from typing import Literal, Optional

import tensorflow as tf
from einops import rearrange

from cvt_tensorflow import __version__
from cvt_tensorflow.models.config import MODELS_CONFIG, TF_WEIGHTS_URL
from cvt_tensorflow.models.layers.utils import (
    Identity_,
    LayerNorm_,
    Linear_,
    TruncNormalInitializer_,
)
from cvt_tensorflow.models.layers.vision_transformer import VisionTransformer
from cvt_tensorflow.models.utils import _to_channel_first


@tf.keras.utils.register_keras_serializable(package="cvt")
class ConvolutionalVisionTransformer(tf.keras.Model):
    def __init__(
        self,
        spec: dict,
        in_chans: int = 3,
        num_classes: int = 1000,
        act_layer: str = "quick_gelu",
        classifier_activation: Optional[str] = None,
        data_format: Literal[
            "channels_first", "channels_last"
        ] = tf.keras.backend.image_data_format(),
        **kwargs,
    ):
        """
        Parameters
        ----------
        spec : dict
            Model specifications.
        in_chans : int, optional
            Number of input channels.
            The default is 3.
        num_classes : int, optional
            Number of classes.
            The default is 1000.
        act_layer : str, optional
            Name of activation layer.
            The default is "quick_gelu".
        classifier_activation : Optional[str], optional
            String name for a tf.keras.layers.Activation layer.
            The default is None.
        data_format : Literal["channels_first", "channels_last"], optional
            A string, one of "channels_last" or "channels_first".
            The ordering of the dimensions in the inputs.
            "channels_last" corresponds to inputs with shape:
            (batch_size, height, width, channels)
            while "channels_first" corresponds to inputs with shape
            (batch_size, channels, height, width).
            The default is tf.keras.backend.image_data_format().
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.spec = spec
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.act_layer = act_layer
        self.classifier_activation = classifier_activation
        self.data_format = data_format

        self.init = self.spec["INIT"]
        self.num_stages = self.spec["NUM_STAGES"]
        in_chans_ = self.in_chans
        for i in range(self.num_stages):
            kwargs = {
                "patch_size": self.spec["PATCH_SIZE"][i],
                "patch_stride": self.spec["PATCH_STRIDE"][i],
                "patch_padding": self.spec["PATCH_PADDING"][i],
                "embed_dim": self.spec["DIM_EMBED"][i],
                "depth": self.spec["DEPTH"][i],
                "num_heads": self.spec["NUM_HEADS"][i],
                "mlp_ratio": self.spec["MLP_RATIO"][i],
                "qkv_bias": self.spec["QKV_BIAS"][i],
                "drop_rate": self.spec["DROP_RATE"][i],
                "attn_drop_rate": self.spec["ATTN_DROP_RATE"][i],
                "drop_path_rate": self.spec["DROP_PATH_RATE"][i],
                "with_cls_token": self.spec["CLS_TOKEN"][i],
                "method": self.spec["QKV_PROJ_METHOD"][i],
                "kernel_size": self.spec["KERNEL_QKV"][i],
                "padding_q": self.spec["PADDING_Q"][i],
                "padding_kv": self.spec["PADDING_KV"][i],
                "stride_kv": self.spec["STRIDE_KV"][i],
                "stride_q": self.spec["STRIDE_Q"][i],
            }
            stage = VisionTransformer(
                in_chans=in_chans_,
                init=self.init,
                act_layer=self.act_layer,
                name=f"stage{i}",
                **kwargs,
            )
            setattr(self, f"stage{i}", stage)
            in_chans_ = self.spec["DIM_EMBED"][i]

        dim_embed = self.spec["DIM_EMBED"][-1]
        self.norm = LayerNorm_(dim_embed, name="norm")
        self.cls_token = self.spec["CLS_TOKEN"][-1]

        # Classifier head
        self.head = (
            Linear_(
                in_features=dim_embed,
                units=self.num_classes,
                bias_initializer="pytorch_uniform",
                kernel_initializer=TruncNormalInitializer_(std=0.02),
                name="head",
            )
            if self.num_classes > 0
            else Identity_(name="head")
        )

        if self.classifier_activation is not None:
            self.classifier_activation_ = tf.keras.layers.Activation(
                self.classifier_activation, dtype=self.dtype, name="pred"
            )

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_token = getattr(self, f"stage{i}")(x)

        if self.cls_token:
            x = self.norm(cls_token)
            x = tf.squeeze(x, axis=[1])
        else:
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.norm(x)
            x = tf.math.reduce_mean(x, axis=1)

        return x

    def call(self, inputs, *args, **kwargs):
        if self.data_format == "channels_last":
            inputs = _to_channel_first(inputs)
        x = self.forward_features(inputs)
        x = self.head(x)
        if hasattr(self, "classifier_activation_"):
            x = self.classifier_activation_(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)

    def __to_functional(self):
        if self.built:
            x = tf.keras.layers.Input(shape=(self._build_input_shape[1:]))
            model = tf.keras.Model(inputs=[x], outputs=self.call(x), name=self.name)
        else:
            raise ValueError(
                "This model has not yet been built. "
                "Build the model first by calling build() or "
                "by calling the model on a batch of data."
            )
        return model

    def summary(self, *args, **kwargs):
        self.__to_functional()
        super().summary(*args, **kwargs)

    def plot_model(self, *args, **kwargs):
        tf.keras.utils.plot_model(model=self.__to_functional(), *args, **kwargs)

    def save(self, *args, **kwargs):
        self.__to_functional().save(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "spec": self.spec,
                "in_chans": self.in_chans,
                "num_classes": self.num_classes,
                "act_layer": self.act_layer,
                "classifier_activation": self.classifier_activation,
                "data_format": self.data_format,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def CvT(
    configuration: Optional[Literal["cvt-13", "cvt-21", "cvt-w24"]] = None,
    pretrained: bool = False,
    pretrained_resolution: Literal[224, 384] = 224,
    pretrained_version: Literal["1k", "22k"] = "1k",
    **kwargs,
) -> tf.keras.Model:
    """Wrapper function for CvT model.

    Parameters
    ----------
    configuration : Optional[Literal["cvt-13", "cvt-21", "cvt-w24"]], optional
        Name of CvT predefined configuration.
        Possible values are: "cvt-13", "cvt-21", "cvt-w24".
        The default is None.
    pretrained : bool, optional
        Whether to use ImageNet pretrained weights.
        The default is False.
    pretrained_resolution : Literal[224, 384], optional
        Image resolution of ImageNet pretrained weights.
        Possible values are: 224, 384.
        The default is 224.
    pretrained_version : Literal["1k", "22k"], optional
        Whether to use ImageNet-1k or ImageNet-22k
        pretrained weights.
        The default is "1k".
    **kwargs
        Additional keyword arguments.

    Raises
    ------
    KeyError
        If choosen configuration not in:
        ["cvt-13","cvt-21","cvt-w24"]

    Returns
    -------
    tf.keras.Model
        CvT model.
    """
    if configuration is not None:
        if configuration in MODELS_CONFIG.keys():
            model = ConvolutionalVisionTransformer(
                spec=MODELS_CONFIG[configuration]["SPEC"],
                name=MODELS_CONFIG[configuration]["name"],
                **kwargs,
            )
            if pretrained:
                if model.data_format == "channels_last":
                    model.build((None, pretrained_resolution, pretrained_resolution, 3))
                elif model.data_format == "channels_first":
                    model.build((None, 3, pretrained_resolution, pretrained_resolution))
                weights_path = "{}/{}/{}-{}x{}_{}.h5".format(
                    TF_WEIGHTS_URL,
                    __version__,
                    configuration,
                    pretrained_resolution,
                    pretrained_resolution,
                    pretrained_version,
                )
                model_weights = tf.keras.utils.get_file(
                    fname="{}-{}x{}_{}.h5".format(
                        configuration,
                        pretrained_resolution,
                        pretrained_resolution,
                        pretrained_version,
                    ),
                    origin=weights_path,
                    cache_subdir="datasets/cvt_tensorflow",
                )
                model.load_weights(model_weights)
            return model
        else:
            raise KeyError(
                f"{configuration} configuration not found. "
                f"Valid values are: {list(MODELS_CONFIG.keys())}"
            )
    else:
        return ConvolutionalVisionTransformer(**kwargs)
