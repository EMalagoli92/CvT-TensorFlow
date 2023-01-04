from typing import Literal

import tensorflow as tf

from cvt_tensorflow.models.layers.attention import Attention_
from cvt_tensorflow.models.layers.mlp import Mlp
from cvt_tensorflow.models.layers.utils import DropPath_, Identity_, LayerNorm_


@tf.keras.utils.register_keras_serializable(package="cvt")
class Block(tf.keras.layers.Layer):
    """Implementation of a Attention MLP block."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        with_cls_token: bool,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "gelu",
        init: Literal["trunc_norm", "xavier"] = "trunc_norm",
        method: Literal["dw_bn", "avg"] = "dw_bn",
        kernel_size: int = 3,
        padding_q: int = 1,
        padding_kv: int = 1,
        stride_kv: int = 1,
        stride_q: int = 1,
        **kwargs
    ):
        """
        Parameters
        ----------
        dim_in : int
            Dimension of input tensor.
        dim_out : int
            Dimension of output tensor.
        num_heads : int
            Number of heads in attention.
        with_cls_token : bool
            Whether to include classification token.
        mlp_ratio : float, optional
            Feature dimension expansion ratio in MLP.
            The default is 4.0.
        qkv_bias : bool, optional
            Whether to use bias in proj_q, proj_k, proj_v.
            The default is False.
        drop : float, optional
            Probability of dropout in convolution projection.
            The default is 0.0.
        attn_drop : float, optional
            Probability of dropout in attention.
            The default is 0.0.
        drop_path : float, optional
            Probability of droppath.
            The default is 0.0.
        act_layer : str, optional
            Name of activation Layer.
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
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.with_cls_token = with_cls_token
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.init = init
        self.method = method
        self.kernel_size = kernel_size
        self.padding_q = padding_q
        self.padding_kv = padding_kv
        self.stride_kv = stride_kv
        self.stride_q = stride_q

    def build(self, input_shape):
        self.norm1 = LayerNorm_(self.dim_in, name="norm1")
        self.attn = Attention_(
            dim_in=self.dim_in,
            dim_out=self.dim_out,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
            init=self.init,
            method=self.method,
            kernel_size=self.kernel_size,
            padding_q=self.padding_q,
            with_cls_token=self.with_cls_token,
            padding_kv=self.padding_kv,
            stride_kv=self.stride_kv,
            stride_q=self.stride_q,
            name="attn",
        )
        self.drop_path_ = (
            DropPath_(self.drop_path, name="drop_path")
            if self.drop_path > 0.0
            else Identity_(name="drop_path")
        )
        self.norm2 = LayerNorm_(self.dim_out, name="norm2")

        dim_mlp_hidden = int(self.dim_out * self.mlp_ratio)
        self.mlp = Mlp(
            in_features=self.dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=self.act_layer,
            drop=self.drop,
            init=self.init,
            name="mlp",
        )
        super().build(input_shape)

    def call(self, inputs, h, w, *args, **kwargs):
        res = inputs

        x = self.norm1(inputs)
        attn = self.attn(x, h, w)
        x = res + self.drop_path_(attn)
        x = x + self.drop_path_(self.mlp(self.norm2(x)))

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "num_heads": self.num_heads,
                "with_cls_token": self.with_cls_token,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "drop": self.drop,
                "attn_drop": self.attn_drop,
                "drop_path": self.drop_path,
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
