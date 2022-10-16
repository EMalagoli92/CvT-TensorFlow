from einops import rearrange
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Type, TypeVar
from functools import partial
from cvt_tensorflow.models.utils import split_
from cvt_tensorflow.models.layers.utils import TruncNormalInitializer_
from cvt_tensorflow.models.layers.utils import LayerNorm_
from cvt_tensorflow.models.layers.conv_embed import ConvEmbed
from cvt_tensorflow.models.layers.block import Block

L = TypeVar("L",bound=tf.keras.layers.Layer)


@tf.keras.utils.register_keras_serializable(package="cvt")
class VisionTransformer(tf.keras.layers.Layer):
    def __init__(self,
                 with_cls_token: bool,
                 patch_size: int = 16,
                 patch_stride: int = 16,
                 patch_padding: int = 0,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 act_layer: Type[L] = partial(tfa.layers.GELU,
                                              approximate = False
                                              ),
                 norm_layer: Type[L] = LayerNorm_,
                 init: str = "trunc_normal",
                 method: str = 'dw_bn',
                 kernel_size: int = 3,
                 padding_q: int = 1,
                 padding_kv: int = 1,
                 stride_kv: int = 1,
                 stride_q: int = 1,
                 **kwargs
                 ):
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
        self.norm_layer = norm_layer
        self.init = init
        self.method = method
        self.kernel_size = kernel_size
        self.padding_q = padding_q
        self.padding_kv = padding_kv
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        
    def build(self,input_shape):       
        self.num_features = self.embed_dim
        self.rearrange = None
        
        self.patch_embed = ConvEmbed(patch_size = self.patch_size,
                                     in_chans = self.in_chans,
                                     stride = self.patch_stride,
                                     padding = self.patch_padding,
                                     embed_dim = self.embed_dim,
                                     norm_layer = self.norm_layer,
                                     name = "patch_embed"
                                     )
        if self.with_cls_token:
            self.cls_token = self.add_weight(name = "cls_token",
                                             shape = (1,1,self.embed_dim),
                                             initializer = TruncNormalInitializer_(std = .02),
                                             trainable = True,
                                             dtype = self.dtype
                                             )
        else:
            self.cls_token = None
        
        self.pos_drop = tf.keras.layers.Dropout(rate = self.drop_rate,
                                                name = "pos_drop"
                                                )
        dpr = [x.numpy() for x in tf.linspace(0.0, self.drop_path_rate, self.depth)]
        
        blocks = []
        for j in range(self.depth):
            blocks.append(
                Block(dim_in = self.embed_dim,
                      dim_out = self.embed_dim,
                      num_heads = self.num_heads,
                      with_cls_token = self.with_cls_token,
                      mlp_ratio = self.mlp_ratio,
                      qkv_bias = self.qkv_bias,
                      drop = self.drop_rate,
                      attn_drop = self.attn_drop_rate,
                      drop_path = dpr[j],
                      act_layer = self.act_layer,
                      norm_layer = self.norm_layer,
                      dense_kernel_initializer = self.init,
                      method = self.method,
                      kernel_size = self.kernel_size,
                      padding_q = self.padding_q,
                      padding_kv = self.padding_kv,
                      stride_kv = self.stride_kv,
                      stride_q = self.stride_q,
                      name = f"blocks/{j}"
                      )
                )
        self.blocks = blocks
        
        super().build(input_shape)
        
    def call(self,inputs,**kwargs):
        x = self.patch_embed(inputs)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        cls_token = None
        if self.cls_token is not None:
            cls_token = tf.broadcast_to(self.cls_token,[tf.shape(x)[0],self.cls_token.shape[1],self.cls_token.shape[2]])
            x = tf.concat([cls_token,x],axis=1)
                                
        x = self.pos_drop(x)
        
        for i, blk in enumerate(self.blocks):
            x = blk(x,H,W)
            
        if self.cls_token is not None:
            cls_token, x = split_(x, [1, H*W], 1)
            
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
         
        return x, cls_token
    
    def get_config(self):
        config = super().get_config()
        config.update({'with_cls_token': self.with_cls_token,
                       'patch_size': self.patch_size,
                       'patch_stride': self.patch_stride,
                       'patch_padding': self.patch_padding,
                       'in_chans': self.in_chans,
                       'embed_dim': self.embed_dim,
                       'depth': self.depth,
                       'num_heads': self.num_heads,
                       'mlp_ratio': self.mlp_ratio,
                       'qkv_bias': self.qkv_bias,
                       'drop_rate': self.drop_rate,
                       'attn_drop_rate': self.attn_drop_rate,
                       'drop_path_rate': self.drop_path_rate,
                       'act_layer': self.act_layer,
                       'norm_layer': self.norm_layer,
                       'init': self.init,
                       'method': self.method,
                       'kernel_size': self.kernel_size,
                       'padding_q': self.padding_q,
                       'padding_kv': self.padding_kv,
                       'stride_kv': self.stride_kv,
                       'stride_q': self.stride_q
                       })
        return config 