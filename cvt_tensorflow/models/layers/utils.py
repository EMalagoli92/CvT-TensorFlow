import math
import warnings
from typing import Type, TypeVar, Union
import tensorflow as tf
import tensorflow_probability as tfp
from cvt_tensorflow.models.utils import _to_channel_last, _to_channel_first


I = TypeVar("I",bound=tf.keras.initializers.Initializer)

def norm_cdf(x):
    '''
    Computes standard normal cumulative distribution.
    '''
    return (1. + math.erf(x / math.sqrt(2.))) / 2.


def trunc_normal_(shape: Union[tf.Tensor,tuple,list], 
                  mean: float = 0., 
                  std: float = 1., 
                  a: float = -2., 
                  b: float = 2.
                  ) -> tf.Tensor:
    '''
    TF2/Keras implementation of timm.models.layers.weight_init.trunc_normal_.
    Create a tf.Tensor filled with values drawn from a truncated normal 
    distribution.
    
    Parameters
    ----------
    shape : tf.Tensor | tuple | list
        Shape of the output tensor.
    mean : float, optional
        The mean of the normal distribution. 
        The default is 0.
    std : float, optional
        The standard deviation of the normal distribution. 
        The default is 1.
    a : float, optional
        The minimum cutoff value.
        The default is -2..
    b : float, optional
        The maximum cutoff value. 
        The default is 2.
    Returns
    -------
    tensor : tf.Tensor
        tf.Tensor filled with values drawn from a truncated normal 
        distribution.
    '''
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)   
    tensor = tf.random.uniform(shape = shape,
                               minval = 2 * l - 1,
                               maxval = 2 * u - 1
                               )
    tensor = tf.math.erfinv(tensor)
    tensor = tf.math.multiply(tensor,std * math.sqrt(2.))
    tensor = tf.math.add(tensor,mean)
    tensor = tf.clip_by_value(tensor,
                              clip_value_min = a,
                              clip_value_max = b
                              )
    return tensor


@tf.keras.utils.register_keras_serializable(package='cvt')
class TruncNormalInitializer_(tf.keras.initializers.Initializer):
    '''
    Initializer version of the previous trunc_normal_.
    '''
    def __init__(self, 
                 mean: float = 0., 
                 std: float = 1., 
                 a: float = -2., 
                 b: float = 2.
                 ):
        '''
        Parameters
        ----------
        mean : float, optional
            The mean of the normal distribution.  
            The default is 0.
        std : float, optional
            The standard deviation of the normal distribution. 
            The default is 1.
        a : float, optional
            The minimum cutoff value. 
            The default is -2.
        b : float, optional
            The maximum cutoff value. 
            The default is 2.
        '''
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b
        
    def __call__(self, shape, dtype = None, **kwargs):
        return trunc_normal_(shape,
                             self.mean,
                             self.std,
                             self.a,
                             self.b)
    
    def get_config(self):
        return {"mean":self.mean, 
                "std": self.std, 
                "a": self.a, 
                "b": self.b}


@tf.keras.utils.register_keras_serializable(package='cvt')
class Dense_(tf.keras.layers.Dense):
    '''
    Custom implementation of tf.keras.layers.Dense with
    kernel_initializer and bias initializer as in the original cvt
    implementation.
    '''
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 kernel_initializer: Union[Type[I],str,None] = None,
                 bias_initializer: Union[Type[I],None] = tf.keras.initializers.Zeros(),
                 **kwargs
                 ):
        '''
        Parameters
        ----------
        in_features : int
            Size of each input sample.
        out_features : int
            Size of each output sample.
        bias : bool, optional
            If set to False, the layer will not learn an additive bias. 
            The default is True.
        kernel_initializer : tf.keras.initializers.Initializer or str, optional
            Initializer for the kernel weights matrix.
            The default is TruncNormalInitializer_(std = .02).
            If None, it will be set to Pytorch Uniform Initializer.
        bias_initializer : tf.keras.initializers.Initializer, optional
            Initializer for the bias vector. 
            The default is tf.keras.initializers.Zeros()
            If None, it will be set to Pytorch Uniform Initializer.
        '''
        self.in_features = in_features

        k = 1/ in_features
        limit = math.sqrt(k)
        uniform_initializer = tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)
        if kernel_initializer == "xavier":
            kernel_initializer = tf.keras.initializers.GlorotUniform()
        else:
            kernel_initializer = TruncNormalInitializer_(std = .02)
            
        if bias_initializer is None:
            bias_initializer = uniform_initializer
        super().__init__(units = out_features,
                         use_bias = bias,
                         kernel_initializer = kernel_initializer,
                         bias_initializer = bias_initializer,
                         **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({'in_features':self.in_features,
                        'out_features':self.units,
                        'kernel_initializer':self.kernel_initializer,
                        'bias_initializer':self.bias_initializer,
                        'bias': self.use_bias})
        return config
    


@tf.keras.utils.register_keras_serializable(package='cvt')
class Conv2d_(tf.keras.layers.Layer):
    '''
    TF2/Keras implementation of torch.nn.Conv2d.
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size : Union[int,tuple,list],
                 stride: Union[int,tuple,list] = 1,
                 padding: int = 0,
                 groups: int = 1,
                 bias: bool = True,
                 kernel_initializer: Union[Type[I],None] = None,
                 bias_initializer: Union[Type[I], None] = None,
                 **kwargs
                 ): 
        '''
        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the convolution.
        kernel_size : Union[int,tuple,list]
            Size of the convolving kernel.
        stride : int | tuple | list, optional
            Stride of the convolution. 
            The default is 1.
        padding : int, optional
            Padding added to all four sides of the input. 
            The default is 0.
        groups : int, optional
            A positive integer specifying the number of groups in which 
            the input is split along the channel axis. 
            The default is 1.
        bias : bool, optional
            If True, adds a learnable bias to the output. 
            The default is True.
        kernel_initializer : tf.keras.initializers.Initializer, optional
            Initializer for the kernel weights matrix. 
            If None, it will be set to Pytorch Uniform Initializer.
        bias_initializer : tf.keras.initializers.Initializer, optional
            Initializer for the bias vector. 
            If None, it will be set to Pytorch Uniform Initializer.
        '''
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups 
        self.bias = bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        # Initializer
        uniform_initializer = self.get_uniform_initializer()
        if self.kernel_initializer is None:
            self.kernel_initializer = uniform_initializer
        if self.bias_initializer is None:
            self.bias_initializer = uniform_initializer

        # Data Format
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.data_format = "channels_first"
        else:
            self.data_format = "channels_last"

        # Pad Layer
        if self.padding > 0:
            self.pad_layer = tf.keras.layers.ZeroPadding2D(padding = padding, 
                                                           data_format = self.data_format)

        self.conv_layer = tf.keras.layers.Conv2D(filters = self.out_channels,
                                                 kernel_size = self.kernel_size,
                                                 use_bias = self.bias,
                                                 padding = 'valid',
                                                 data_format=self.data_format,
                                                 groups = self.groups,
                                                 kernel_initializer = self.kernel_initializer,
                                                 bias_initializer = self.bias_initializer,
                                                 strides = self.stride,
                                                 )   

    def get_uniform_initializer(self):
        # Uniform Initializer
        if isinstance(self.kernel_size, int):
            kernel_product = self.kernel_size**2
        else:
            kernel_product = self.kernel_size[0] * self.kernel_size[1]
        k = self.groups / (self.in_channels * kernel_product)
        limit = math.sqrt(k)
        uniform_initializer = tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)
        return uniform_initializer

    def call(self,inputs,**kwargs):
        if self.data_format == "channels_last":
            inputs = _to_channel_last(inputs)
        if self.padding > 0:
            x = self.pad_layer(inputs)
        else:
            x = inputs
        x = self.conv_layer(x)
        if self.data_format == "channels_last":
            x = _to_channel_first(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'in_channels':self.in_channels,
                       'out_channels':self.out_channels,
                       'kernel_size':self.kernel_size,
                       'stride':self.stride,
                       'padding':self.padding,
                       'groups':self.groups, 
                       'bias': self.bias,
                       'kernel_initializer': self.kernel_initializer,
                       'bias_initializer': self.bias_initializer
                       })
        return config
    

@tf.keras.utils.register_keras_serializable(package='cvt')
class BatchNorm2d_(tf.keras.layers.BatchNormalization):
    '''
    TF2/Keras implementation of torch.nn.BatchNorm2d.
    '''
    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**kwargs,
                         epsilon = 1e-05,
                         momentum = .9,
                         axis = 1
                         )

    def get_config(self):
        config = super().get_config()
        return config
    
    
@tf.keras.utils.register_keras_serializable(package='cvt')
class AvgPool2d_(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size: Union[int,tuple],
                 stride: Union[int,tuple,None] = None,
                 padding: Union[int,tuple] = 0,
                 ceil_mode: bool = False,
                 **kwargs
                 ):
        '''
        Parameters
        ----------
        kernel_size : Union[int,tuple]
            The size of the window
        stride : Optional[int,tuple], optional
            The stride of the window. 
            The default is kernel_size 
        padding : Union[int,tuple], optional
            Implicit zero padding to be added on both sides. 
            The default is 0.
        ceil_mode : bool, optional
            When True, will use ceil instead of floor to compute the output 
            shape. 
            The default is False.
            
        Note
        - kernel_size, stride, padding can be:
                - a single int – in which case the same value is used for the 
                  height and width dimension.
                - a tuple of two ints – in which case, the first int is used 
                  for the height dimension, and the second int for the width 
                  dimension.        
         - When ceil_mode=True, sliding windows are allowed to go off-bounds 
           if they start within the left padding or the input. 
           Sliding windows that would start in the right padded region are 
           ignored.
            
        '''
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode        
        
        # Default value for value is kernel_size
        if self.stride is None:
            self.stride = kernel_size
        
        
        if isinstance(self.kernel_size,int):
            self.kernel_size = (self.kernel_size,self.kernel_size)    
        if isinstance(self.stride,int):
            self.stride = (self.stride,self.stride)
        if isinstance(self.padding,int):
            self.padding = (self.padding,self.padding)
            
        # Data Format
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.data_format = "channels_first"
        else:
            self.data_format = "channels_last"     
            
            
        # pad Layer
        self.pad_layer = tf.keras.layers.ZeroPadding2D(padding = self.padding, 
                                                       data_format = "channels_first")
        
        # avg pool layer
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size = self.kernel_size,
                                                         strides = self.stride,
                                                         padding = "valid",
                                                         data_format = self.data_format
                                                         )
        
    
    def prepare(self,x):
        B, C, H_in, W_in = x.shape

        h_out = (H_in - self.kernel_size[0]) / self.stride[0] +1
        w_out = (W_in - self.kernel_size[1]) / self.stride[1] +1
        # Output shape ceil mode
        H_out_ceil = math.ceil(h_out)
        W_out_ceil = math.ceil(w_out)

        # Output shape floor mode
        H_out_floor = math.floor(h_out)
        W_out_floor = math.floor(w_out)
        
        start_new_kernel_index = (H_out_floor * self.stride[0],W_out_floor * self.stride[1])
        
        if (H_out_ceil != H_out_floor):
            n_in_rows = H_in - start_new_kernel_index[0]
            n_missing_rows = (self.kernel_size[0] - n_in_rows)
            
            if (start_new_kernel_index[0] <= H_in - self.padding[0] -1):
                new_row = (n_missing_rows / n_in_rows) * tf.math.reduce_sum(input_tensor = x[:,:,start_new_kernel_index[0]:,:],
                                                                            axis=[2],
                                                                            keepdims=True
                                                                            )
                zeros = tf.tile(tf.zeros(new_row.shape),[1,1,n_missing_rows-1,1])
                x = tf.concat([x,new_row,zeros],axis=2)
            
        if (W_out_ceil != W_out_floor):
            n_in_columns = W_in - start_new_kernel_index[1]
            n_missing_columns = (self.kernel_size[1] - n_in_columns)
            
            if (start_new_kernel_index[1] <= W_in - self.padding[1] -1):
                new_column =  (n_missing_columns / n_in_columns) * tf.math.reduce_sum(input_tensor = x[:,:,:,start_new_kernel_index[1]:],
                                                                                      axis=[3],
                                                                                      keepdims=True
                                                                                      )
                zeros = tf.tile(tf.zeros(new_column.shape),[1,1,1,n_missing_columns-1])
                x = tf.concat([x,new_column,zeros],axis=3)
        return x     
        
    
    def call(self,inputs,**kwargs):
        x = self.pad_layer(inputs)
        if self.ceil_mode:
            x = self.prepare(x)
        if self.data_format == 'channels_last':
            x = _to_channel_last(x)
        x = self.avg_pool(x)
        if self.data_format == "channels_last":
            x = _to_channel_first(x)
        return x
            
            
    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size,
                       'stride': self.stride,
                       'padding': self.padding,
                       'ceil_mode': self.ceil_mode
                       })
        return config
    
    
@tf.keras.utils.register_keras_serializable(package='cvt')
class LayerNorm_(tf.keras.layers.LayerNormalization):
    '''
    TF2/Keras implementation of torch.nn.LayerNorm.
    '''
    def __init__(self,
                 normalized_shape: Union[int,tuple],
                 **kwargs):
        '''
        Parameters
        ----------
        normalized_shape : Union[int,tuple]
            input shape from an expected input of size
            [∗ x normalized_shape[0] x normalized_shape[1] x ... xnormalized_shape[−1]]
            If a single integer is used, it is treated as a singleton list, 
            and this module will normalize over the last dimension which is 
            expected to be of that specific size.
        '''
        super().__init__(**kwargs,
                         epsilon = 1e-5)
        self.normalized_shape = normalized_shape

    def build(self,input_shape):

        if isinstance(self.normalized_shape,int):
            self.lnm = 1
        if isinstance(self.normalized_shape,tuple):
            self.lnm = len(self.normalized_shape)
        self.axis = tuple(range(input_shape.rank - self.lnm,input_shape.rank))
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({'normalized_shape':self.normalized_shape})
        return config

    
@tf.keras.utils.register_keras_serializable(package='cvt')    
class Identity_(tf.keras.layers.Layer):
    '''
    TF2/Keras implementation of torch.nn.Identity.
    '''
    def __init__(self,*args,**kwargs):
        super().__init__(**kwargs)

    def call(self,inputs):
        return inputs

    def get_config(self):
        config = super().get_config()
        return config   
    

def drop_path(x, 
              drop_prob: float = 0., 
              training: bool = False, 
              scale_by_keep: bool = True):
    '''
    TF2/Keras implementation of timm.models.layers.drop_path.
    Drop paths (Stochastic Depth) per sample (when applied in main path 
    of residual blocks)
    '''
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    _shape1_ = tf.expand_dims([tf.shape(x)[0]],1)
    _shape2_ = tf.expand_dims(tf.ones((tf.rank(x)-1),dtype=tf.int32),0)
    shape = tf.reshape(tf.concat([_shape1_,_shape2_],1),[-1]) 
    random_tensor = tfp.distributions.Bernoulli(probs = keep_prob,dtype=x.dtype).sample(sample_shape = shape)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = tf.math.divide(random_tensor,keep_prob)
    return x * random_tensor


@tf.keras.utils.register_keras_serializable(package='cvt')    
class DropPath_(tf.keras.layers.Layer):
    '''
    TF2/Keras implementation of timm.models.layers.DropPath.
    Drop paths (Stochastic Depth) per sample  (when applied in main path 
    of residual blocks).
    '''
    def __init__(self,
                 drop_prob: float = 0.,
                 scale_by_keep: bool = True,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep


    def call(self,inputs,training=None,**kwargs):
        return drop_path(inputs,self.drop_prob, training, self.scale_by_keep)

    def get_config(self):
        config = super().get_config()
        config.update({'drop_prob':self.drop_prob,
                       'scale_by_keep':self.scale_by_keep})
        return config
    
    
@tf.keras.utils.register_keras_serializable(package='cvt')
class QuickGELU_(tf.keras.layers.Layer):
    '''
    TF2/Keras implementation of QuickGELU present in the original cvt
    implementation.
    '''
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def call(self,inputs,**kwargs):
        return inputs * tf.keras.layers.Activation(tf.nn.sigmoid)(1.702 * inputs)
        
    def get_config(self):
        config = super().get_config()
        return config