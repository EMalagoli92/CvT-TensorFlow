from einops import rearrange
import tensorflow as tf
from typing import Union, Type, TypeVar
from cvt_tensorflow.models.utils import to_2tuple
from cvt_tensorflow.models.layers.utils import Conv2d_

L = TypeVar("L",bound=tf.keras.layers.Layer)


@tf.keras.utils.register_keras_serializable(package='cvt')
class ConvEmbed(tf.keras.layers.Layer):
    def __init__(self,
                 patch_size: int = 7,
                 in_chans: int = 3,
                 embed_dim: int = 64,
                 stride: int = 4,
                 padding: int = 2,
                 norm_layer: Union[Type[L],None] = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.stride = stride
        self.padding = padding
        self.norm_layer = norm_layer
        
    def build(self,input_shape):
        patch_size = to_2tuple(self.patch_size)
        
        self.proj = Conv2d_(in_channels = self.in_chans,
                            out_channels = self.embed_dim,
                            kernel_size = patch_size,
                            stride = self.stride,
                            padding = self.padding,
                            name = "proj"
                            )
        
        self.norm = self.norm_layer(self.embed_dim, name = "norm")\
                    if self.norm_layer else None
                                                    
        super().build(input_shape)
        
    def call(self,inputs,**kwargs):
        x = self.proj(inputs)
        
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x        
    
    def get_config(self):
        config = super().get_config()
        config.update({'patch_size': self.patch_size,
                       'in_chans': self.in_chans,
                       'embed_dim': self.embed_dim,
                       'stride': self.stride,
                       'padding': self.padding,
                       'norm_layer': self.norm_layer
                       })
        return config  