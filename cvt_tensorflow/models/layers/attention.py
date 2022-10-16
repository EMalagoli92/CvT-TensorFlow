import numpy as np
import tensorflow as tf
from einops import rearrange
from einops.layers.tensorflow import Rearrange
from typing import Union, Type, TypeVar
from cvt_tensorflow.models.utils import split_, run_layers_list
from cvt_tensorflow.models.layers.utils import Conv2d_, AvgPool2d_,\
                                               BatchNorm2d_, Dense_

I = TypeVar("I",bound=tf.keras.initializers.Initializer)


@tf.keras.utils.register_keras_serializable(package='cvt')
class Attention(tf.keras.layers.Layer):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 num_heads: int,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 method: str = 'dw_bn',
                 kernel_size: int = 3,
                 stride_kv: int = 1,
                 stride_q: int = 1,
                 padding_kv: int = 1,
                 padding_q: int = 1,
                 with_cls_token: bool = True,
                 dense_kernel_initializer: Union[Type[I],str,None] = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.method = method
        self.kernel_size = kernel_size
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.padding_kv = padding_kv
        self.padding_q = padding_q
        self.with_cls_token = with_cls_token
        self.dense_kernel_initializer = dense_kernel_initializer
        
    def build(self,input_shape):
        self.dim = self.dim_out
        self.scale = self.dim_out ** (-0.5)
        self.conv_proj_q = self._build_projection(dim_in = self.dim_in, 
                                                  dim_out = self.dim_out, 
                                                  kernel_size = self.kernel_size, 
                                                  padding = self.padding_q, 
                                                  stride = self.stride_q, 
                                                  method = "linear" if self.method == "avg" else self.method,
                                                  name = "conv_proj_q"
                                                  )
        self.conv_proj_k = self._build_projection(dim_in = self.dim_in,
                                                  dim_out = self.dim_out,
                                                  kernel_size = self.kernel_size,
                                                  padding = self.padding_kv,
                                                  stride = self.stride_kv,
                                                  method = self.method,
                                                  name = "conv_proj_k"
                                                  )
        self.conv_proj_v = self._build_projection(dim_in = self.dim_in,
                                                  dim_out = self.dim_out,
                                                  kernel_size = self.kernel_size,
                                                  padding = self.padding_kv,
                                                  stride = self.stride_kv,
                                                  method = self.method,
                                                  name = "conv_proj_v"
                                                  )
        self.proj_q = Dense_(in_features = self.dim_in,
                             out_features = self.dim_out,
                             bias = self.qkv_bias,
                             kernel_initializer = self.dense_kernel_initializer,
                             name = "proj_q"
                             )
        self.proj_k = Dense_(in_features = self.dim_in,
                             out_features = self.dim_out,
                             bias = self.qkv_bias,
                             kernel_initializer = self.dense_kernel_initializer,
                             name = "proj_k"
                             )
        self.proj_v = Dense_(in_features = self.dim_in,
                             out_features = self.dim_out,
                             bias = self.qkv_bias,
                             kernel_initializer = self.dense_kernel_initializer,
                             name = "proj_v"
                             )
        self.attn_drop = tf.keras.layers.Dropout(rate = self.attn_drop, 
                                                 name = "attn_drop"
                                                 )
        self.proj = Dense_(in_features = self.dim_out,
                           out_features = self.dim_out,
                           kernel_initializer = self.dense_kernel_initializer,
                           name = "proj"
                           )
        self.proj_drop = tf.keras.layers.Dropout(rate = self.proj_drop,
                                                 name = "proj_drop"
                                                 )
        
        super().build(input_shape)
        
    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method,
                          name
                          ):
        if method == "dw_bn":
            proj = [Conv2d_(in_channels = dim_in,
                            out_channels = dim_in,
                            kernel_size = kernel_size,
                            stride = stride,
                            padding = padding,
                            bias = False,
                            groups = dim_in,
                            name = f"{name}/conv"
                            ),
                    BatchNorm2d_(name = f"{name}/bn"),
                    Rearrange('b c h w -> b (h w) c')
                    ]
        elif method == "avg":
            proj = [AvgPool2d_(kernel_size = kernel_size,
                               padding = padding,
                               stride = stride,
                               ceil_mode = True,
                               name = f"{name}/avg"
                               ),
                    Rearrange('b c h w -> b (h w) c')
                    ]
        elif method == "linear":
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))
            
        return proj
        
    def forward_conv(self,
                     x,
                     h,
                     w):
        if self.with_cls_token:
            cls_token, x = split_(tensor = x,
                                  split_size_or_sections = [1, h*w],
                                  dim = 1
                                  )
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        if self.conv_proj_q is not None:
            q = run_layers_list(x,self.conv_proj_q)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')
        if self.conv_proj_k is not None:
            k = run_layers_list(x,self.conv_proj_k)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')
        if self.conv_proj_v is not None:
            v = run_layers_list(x,self.conv_proj_v)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')
        if self.with_cls_token:
            q = tf.concat([cls_token, q],axis=1)
            k = tf.concat([cls_token, k],axis=1)
            v = tf.concat([cls_token, v],axis=1)            
            
        return q, k, v
            
    def call(self,
             inputs,
             h,
             w,
             **kwargs):
        if (self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
            ):
            q, k, v = self.forward_conv(inputs, h, w)
            
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        
        attn_score = tf.einsum('bhlk,bhtk->bhlt', q, k) * self.scale
        attn = tf.keras.activations.softmax(attn_score,axis=-1)
        attn = self.attn_drop(attn)
        
        x = tf.einsum('bhlt,bhtv->bhlv', attn, v)
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x        
    
    @staticmethod
    def compute_macs(module,
                     input,
                     output
                     ):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0
        
        _, T, C = input.shape
        H = W = int(np.sqrt(T-1)) if module.with_cls_token else int(np.sqrt(T))
        
        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV        
        
        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV
        
        for suffix in ['q','k','v']:
            if hasattr(module,f"conv_proj_{suffix}"):
                params = {layer.name:tf.math.reduce_sum([tf.size(p,tf.float32) for p in layer.weights]) for layer in getattr(module,f"conv_proj_{suffix}",0)\
                                                                                                        if layer.name == f"conv_proj_{suffix}/conv"}
                params = params.get(f"conv_proj_{suffix}/conv",0)
                if suffix == 'q':
                    flops += params * H_Q * W_Q
                else:
                    flops += params * H_KV * W_KV
            
        params = sum([tf.size(p,tf.float32) for p in module.proj_q.weights])
        flops += params * T_Q
        params = sum([tf.size(p,tf.float32) for p in module.proj_k.weights])
        flops += params * T_KV
        params = sum([tf.size(p,tf.float32) for p in module.proj_v.weights])
        flops += params * T_KV
        params = sum([tf.size(p,tf.float32) for p in module.proj.weights])
        flops += params * T
        
        return flops
            
    def get_config(self):
        config = super().get_config()
        config.update({'dim_in': self.dim_in,
                       'dim_out': self.dim_out,
                       'num_heads': self.num_heads,
                       'qkv_bias': self.qkv_bias,
                       'attn_drop': self.attn_drop,
                       'proj_drop': self.proj_drop,
                       'method': self.method,
                       'kernel_size': self.kernel_size,
                       'stride_kv': self.stride_kv,
                       'stride_q': self.stride_q,
                       'padding_kv': self.padding_kv,
                       'padding_q': self.padding_q,
                       'with_cls_token': self.with_cls_token,
                       'dense_kernel_initializer': self.dense_kernel_initializer
                       })
        return config