import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial
from typing import Union, Type, TypeVar
from cvt_tensorflow.models.layers.utils import LayerNorm_, DropPath_, Identity_
from cvt_tensorflow.models.layers.attention import Attention
from cvt_tensorflow.models.layers.mlp import Mlp

I = TypeVar("I",bound=tf.keras.initializers.Initializer)
L = TypeVar("L",bound=tf.keras.layers.Layer)


@tf.keras.utils.register_keras_serializable(package='cvt')
class Block(tf.keras.layers.Layer):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 num_heads: int,
                 with_cls_token: bool,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: Type[L] = partial(tfa.layers.GELU,
                                              approximate = False
                                              ),
                 norm_layer: Type[L] = LayerNorm_,
                 dense_kernel_initializer: Union[Type[I],str,None] = None,
                 method: str = 'dw_bn',
                 kernel_size: int = 3,
                 padding_q: int = 1,
                 padding_kv: int = 1,
                 stride_kv: int = 1,
                 stride_q: int = 1,
                 **kwargs
                 ):
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
        self.norm_layer = norm_layer
        self.dense_kernel_initializer = dense_kernel_initializer
        self.method = method
        self.kernel_size = kernel_size
        self.padding_q = padding_q
        self.padding_kv = padding_kv
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        
    def build(self,input_shape):
        self.norm1 = self.norm_layer(self.dim_in, name = "norm1")
        self.attn = Attention(dim_in = self.dim_in,
                              dim_out = self.dim_out,
                              num_heads = self.num_heads,
                              qkv_bias = self.qkv_bias,
                              attn_drop = self.attn_drop,
                              proj_drop = self.drop,
                              dense_kernel_initializer = self.dense_kernel_initializer,
                              method = self.method,
                              kernel_size = self.kernel_size,
                              padding_q = self.padding_q,
                              with_cls_token = self.with_cls_token,
                              padding_kv = self.padding_kv,
                              stride_kv = self.stride_kv,
                              stride_q = self.stride_q,
                              name = "attn"
                              )
        self.drop_path_ = DropPath_(self.drop_path,name = "drop_path")\
                         if self.drop_path > 0. else Identity_(name = "drop_path")
        self.norm2 = self.norm_layer(self.dim_out, name = "norm2")
        
        dim_mlp_hidden = int(self.dim_out * self.mlp_ratio)
        self.mlp = Mlp(in_features = self.dim_out,
                       hidden_features = dim_mlp_hidden,
                       act_layer = self.act_layer,
                       drop = self.drop,
                       dense_kernel_initializer = self.dense_kernel_initializer,
                       name = "mlp"
                       )
        super().build(input_shape)
        
    def call(self,
             inputs,
             h,
             w,
             **kwargs
             ):
        res = inputs
        
        x = self.norm1(inputs)
        attn = self.attn(x,h,w)
        x = res + self.drop_path_(attn)
        x = x + self.drop_path_(self.mlp(self.norm2(x)))
        
        return x
        
    def get_config(self):
        config = super().get_config()
        config.update({'dim_in': self.dim_in,
                       'dim_out': self.dim_out,
                       'num_heads': self.num_heads,
                       'with_cls_token': self.with_cls_token,
                       'mlp_ratio': self.mlp_ratio,
                       'qkv_bias': self.qkv_bias,
                       'drop': self.drop,
                       'attn_drop': self.attn_drop,
                       'drop_path': self.drop_path,
                       'act_layer': self.act_layer,
                       'norm_layer': self.norm_layer,
                       'dense_kernel_initializer': self.dense_kernel_initializer,
                       'method': self.method,
                       'kernel_size': self.kernel_size,
                       'padding_q': self.padding_q,
                       'padding_kv': self.padding_kv,
                       'stride_kv': self.stride_kv,
                       'stride_q': self.stride_q
                       })
        return config