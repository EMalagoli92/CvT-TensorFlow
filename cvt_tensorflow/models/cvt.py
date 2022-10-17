import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import numpy as np
import json
from einops import rearrange
from typing import Optional, Type, TypeVar
from functools import partial
import tensorflow_addons as tfa
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.python.framework import random_seed
from cvt_tensorflow.models.config import MODELS_CONFIG, TF_WEIGHTS_URL
from cvt_tensorflow.models.utils import _to_channel_first
from cvt_tensorflow.models.layers.utils import LayerNorm_, Dense_, Identity_, \
                                               QuickGELU_
from cvt_tensorflow.models.layers.vision_transformer import VisionTransformer

L = TypeVar("L",bound=tf.keras.layers.Layer)

# Set Seed
SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
random_seed.set_seed(SEED)
np.random.seed(SEED)

version_path = os.path.normpath(os.path.join(os.path.split(os.path.dirname(__file__))[0],"version.json"))
with open(version_path,'r') as handle:
    VERSION = json.load(handle)['VERSION'] 


class ConvolutionalVisionTransformer(tf.keras.Model):
    def __init__(self,
                 in_chans: int =3,
                 num_classes: int =1000,
                 act_layer: Type[L] = QuickGELU_,
                 norm_layer: Type[L] = LayerNorm_,
                 init: str ="trunc_norm",
                 spec: Optional[dict] = None,
                 classifier_activation: Optional[str] = None,
                 data_format: Optional[str] = tf.keras.backend.image_data_format(),
                 **kwargs
                 ):
        super().__init__(**kwargs)
        if act_layer == tfa.layers.GELU:
            act_layer = partial(tfa.layers.GELU,
                                approximate = False
                                )
        self.num_classes = num_classes
        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {'patch_size': spec['PATCH_SIZE'][i],
                      'patch_stride': spec['PATCH_STRIDE'][i],
                      'patch_padding': spec['PATCH_PADDING'][i],
                      'embed_dim': spec['DIM_EMBED'][i],
                      'depth': spec['DEPTH'][i],
                      'num_heads': spec['NUM_HEADS'][i],
                      'mlp_ratio': spec['MLP_RATIO'][i],
                      'qkv_bias': spec['QKV_BIAS'][i],
                      'drop_rate': spec['DROP_RATE'][i],
                      'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                      'drop_path_rate': spec['DROP_PATH_RATE'][i],
                      'with_cls_token': spec['CLS_TOKEN'][i],
                      'method': spec['QKV_PROJ_METHOD'][i],
                      'kernel_size': spec['KERNEL_QKV'][i],
                      'padding_q': spec['PADDING_Q'][i],
                      'padding_kv': spec['PADDING_KV'][i],
                      'stride_kv': spec['STRIDE_KV'][i],
                      'stride_q': spec['STRIDE_Q'][i],
                      }
            stage = VisionTransformer(in_chans = in_chans,
                                      init = init,
                                      act_layer = act_layer,
                                      norm_layer = norm_layer,
                                      name = f"stage{i}",
                                      **kwargs
                                      )
            setattr(self, f"stage{i}",stage)
            in_chans = spec['DIM_EMBED'][i]
            
        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed, name = "norm")
        self.cls_token = spec['CLS_TOKEN'][-1]
        
        # Classifier head
        self.head = Dense_(in_features = dim_embed,
                           out_features = num_classes,
                           bias_initializer = None,
                           name = "head"
                           ) if num_classes > 0 else Identity_(name = "head")
        
        if classifier_activation is not None:
            self.classifier_activation = tf.keras.layers.Activation(classifier_activation,
                                                                    dtype = self.dtype,
                                                                    name = "pred"
                                                                    )
        self.data_format = data_format
        
    def forward_features(self,x):
        for i in range(self.num_stages):
            x, cls_token = getattr(self,f"stage{i}")(x)
            
        if self.cls_token:
            x = self.norm(cls_token)
            x = tf.squeeze(x,axis=[1])
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = tf.math.reduce_mean(x,axis=1)
            
        return x
    
    
    def call(self,inputs,**kwargs):
        if self.data_format == "channels_last":
            inputs = _to_channel_first(inputs)
        x = self.forward_features(inputs)
        x = self.head(x)
        if hasattr(self, "classifier_activation"):
            x = self.classifier_activation(x)
        return x
    
    
def CvT(configuration: Optional[str] = None,
        pretrained: bool = False,
        pretrained_resolution: int = 224,
        pretrained_version: str = '1k',
        **kwargs
        ) -> tf.keras.Model:
    '''
    Wrapper function for CvT model.

    Parameters
    ----------
    configuration : Optional[str], optional
        Name of CvT predefined configuration. 
        Possible values are: cvt-13, cvt-21, cvt-w24
        The default is None.
    pretrained : bool, optional
        Whether to use ImageNet pretrained weights. 
        The default is False.
    pretrained_resolution : int, optional
        Image resolution of ImageNet pretrained weights.
        Possible values are: 224, 384
        The default is 224.
    pretrained_version : str, optional
        Whether to use ImageNet-1k or ImageNet-22k 
        pretrained weights.
        The default is '1k'.

    Raises
    ------
    KeyError
        If choosen configuration not in:
            ['cvt-13','cvt-21','cvt-w24']

    Returns
    -------
    CvT model (tf.keras.Model)
    '''
    if configuration is not None:
        if configuration in MODELS_CONFIG.keys():
            model = ConvolutionalVisionTransformer(spec = MODELS_CONFIG[configuration]['SPEC'],
                                                   name = MODELS_CONFIG[configuration]['name'],
                                                   **kwargs
                                                   )
            if pretrained:
                if model.data_format == "channels_last":
                    model(tf.ones((1,pretrained_resolution,pretrained_resolution,3)))
                elif model.data_format == "channels_first":
                    model(tf.ones((1,3,pretrained_resolution,pretrained_resolution)))
                weights_path = "{}/{}/{}-{}x{}_{}.h5".format(TF_WEIGHTS_URL,VERSION,
                                                             configuration,
                                                             pretrained_resolution,
                                                             pretrained_resolution,
                                                             pretrained_version
                                                             )
                model_weights = tf.keras.utils.get_file(fname = "{}-{}x{}_{}.h5".format(configuration,
                                                                                        pretrained_resolution,
                                                                                        pretrained_resolution,
                                                                                        pretrained_version
                                                                                        ),
                                                        origin = weights_path,
                                                        cache_subdir = "datasets/cvt_tensorflow"
                                                        )
                model.load_weights(model_weights)
            return model
        else:
            raise KeyError(f"{configuration} configuration not found. Valid values are: {list(MODELS_CONFIG.keys())}")
    else:
        return ConvolutionalVisionTransformer(**kwargs)