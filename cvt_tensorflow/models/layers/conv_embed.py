import tensorflow as tf
from einops import rearrange

from cvt_tensorflow.models.layers.utils import Conv2d_, LayerNorm_
from cvt_tensorflow.models.utils import _ntuple


@tf.keras.utils.register_keras_serializable(package="cvt")
class ConvEmbed(tf.keras.layers.Layer):
    """Projects image patches into embedding space using multiple Convolution
    and maxpooling layers."""

    def __init__(
        self,
        patch_size: int = 7,
        in_chans: int = 3,
        embed_dim: int = 64,
        stride: int = 4,
        padding: int = 2,
        **kwargs
    ):
        """
        Parameters
        ----------
        patch_size : int, optional
            Size of a patch.
            The default is 7.
        in_chans : int, optional
            Number of input channels.
            The default is 3.
        embed_dim : int, optional
            Dimension of hidden layer.
            The default is 64.
        stride : int, optional
            Stride of the convolution operation.
            The default is 4.
        padding : int, optional
            Padding to all sides of the input.
            The default is 2.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.stride = stride
        self.padding = padding

    def build(self, input_shape):
        patch_size = _ntuple(2)(self.patch_size)

        self.proj = Conv2d_(
            in_channels=self.in_chans,
            out_channels=self.embed_dim,
            kernel_size=patch_size,
            stride=self.stride,
            padding=self.padding,
            name="proj",
        )

        self.norm = LayerNorm_(self.embed_dim, name="norm")

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.proj(inputs)

        x_shape = tf.shape(x)
        H = x_shape[2]
        W = x_shape[3]
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm:
            x = self.norm(x)
        x = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, tf.shape(x)[2], H, W])

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "in_chans": self.in_chans,
                "embed_dim": self.embed_dim,
                "stride": self.stride,
                "padding": self.padding,
            }
        )
        return config
