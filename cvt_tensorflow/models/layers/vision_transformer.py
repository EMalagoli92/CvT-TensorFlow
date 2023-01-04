from typing import Literal

import tensorflow as tf
from einops import rearrange

from cvt_tensorflow.models.layers.block import Block
from cvt_tensorflow.models.layers.conv_embed import ConvEmbed
from cvt_tensorflow.models.layers.utils import TruncNormalInitializer_


@tf.keras.utils.register_keras_serializable(package="cvt")
class VisionTransformer(tf.keras.layers.Layer):
    """Vision Transformer with support for patch or hybrid CNN input stage."""

    def __init__(
        self,
        with_cls_token: bool,
        patch_size: int = 16,
        patch_stride: int = 16,
        patch_padding: int = 0,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: str = "gelu",
        init: Literal["trunc_norm", "xavier"] = "trunc_norm",
        method: Literal["dw_bn", "avg"] = "dw_bn",
        kernel_size: int = 3,
        padding_q: int = 1,
        padding_kv: int = 1,
        stride_kv: int = 1,
        stride_q: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        with_cls_token : bool
            Whether to include classification token.
        patch_size : int, optional
            Size of patch.
            The default is 16.
        patch_stride : int, optional
            Stride of patch.
            The default is 16.
        patch_padding : int, optional
            Padding for patch.
            The default is 0.
        in_chans : int, optional
            Number of input channels.
            The default is 3.
        emebd_dim : int, optional
            Embedding dimension.
            The default is 768.
        depth : int, optional
            Number of CVT Attention blocks in each stage.
            The default is 12.
        num_heads : int, optional
            Number of heads in attention.
            The default is 12.
        mlp_ratio : float, optional
            Feature dimension expansion ratio in MLP.
            The default is 4.0.
        qkv_bias : bool, optional
            Whether to use bias in proj_q, proj_k, proj_v.
            The default is False.
        drop_rate : float, optional
            Probability of dropout in convolution projection.
            The default is 0.0.
        attn_drop_rate : float, optional
            Probability of dropout in attention.
            The default is 0.0.
        drop_path_rate : float, optional
            Probability for droppath.
            The default is 0.0.
        act_layer : str, optional
            The activation layer.
            The default is "gelu".
        init : Literal["trunc_norm", "xavier"], optional
            Initialization method.
            Possible values are: "trunc_norm", "xavier".
            The default is "trunc_norm".
        method : Literal["dw_bn", "avg"], optional
            Method of projection, "dw_bn" for depth-wise convolution
            and batch norm, "avg" for average pooling.
            The default is "dw_bn".
        kernel_size : int, optional
            Size of kernel.
            The default is 3.
        padding_q : int, optional
            Padding for query.
            The default is 1.
        padding_kv : int, optional
            Padding for key value.
            The default is 1.
        stride_kv : int, optional
            Size of stride for key value.
            The default is 1.
        stride_q : int, optional
            Size of stride for query.
            The default is 1.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.with_cls_token = with_cls_token
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.act_layer = act_layer
        self.init = init
        self.method = method
        self.kernel_size = kernel_size
        self.padding_q = padding_q
        self.padding_kv = padding_kv
        self.stride_kv = stride_kv
        self.stride_q = stride_q

    def build(self, input_shape):
        self.num_features = self.embed_dim
        self.rearrange = None

        self.patch_embed = ConvEmbed(
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            stride=self.patch_stride,
            padding=self.patch_padding,
            embed_dim=self.embed_dim,
            name="patch_embed",
        )
        if self.with_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.embed_dim),
                initializer=TruncNormalInitializer_(std=0.02),
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.cls_token = None

        self.pos_drop = tf.keras.layers.Dropout(rate=self.drop_rate, name="pos_drop")
        dpr = [
            i * self.drop_path_rate / (self.depth - 1) if self.depth > 1 else 0.0
            for i in range(self.depth)
        ]

        blocks = []
        for j in range(self.depth):
            blocks.append(
                Block(
                    dim_in=self.embed_dim,
                    dim_out=self.embed_dim,
                    num_heads=self.num_heads,
                    with_cls_token=self.with_cls_token,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop=self.drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=self.act_layer,
                    init=self.init,
                    method=self.method,
                    kernel_size=self.kernel_size,
                    padding_q=self.padding_q,
                    padding_kv=self.padding_kv,
                    stride_kv=self.stride_kv,
                    stride_q=self.stride_q,
                    name=f"blocks/{j}",
                )
            )
        self.blocks = blocks

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.patch_embed(inputs)
        x_shape = tf.shape(x)
        B = x_shape[0]
        H = x_shape[2]
        W = x_shape[3]
        x = rearrange(x, "b c h w -> b (h w) c")

        cls_token = None
        if self.cls_token is not None:
            cls_token_shape = tf.shape(self.cls_token)
            cls_token = tf.broadcast_to(
                self.cls_token, [B, cls_token_shape[1], cls_token_shape[2]]
            )
            x = tf.concat([cls_token, x], axis=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_token, x = tf.split(value=x, num_or_size_splits=[1, H * W], axis=1)
        x = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, tf.shape(x)[2], H, W])

        return x, cls_token

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "with_cls_token": self.with_cls_token,
                "patch_size": self.patch_size,
                "patch_stride": self.patch_stride,
                "patch_padding": self.patch_padding,
                "in_chans": self.in_chans,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "drop_rate": self.drop_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "act_layer": self.act_layer,
                "init": self.init,
                "method": self.method,
                "kernel_size": self.kernel_size,
                "padding_q": self.padding_q,
                "padding_kv": self.padding_kv,
                "stride_kv": self.stride_kv,
                "stride_q": self.stride_q,
            }
        )
        return config
