import tensorflow as tf
import tensorflow_addons as tfa
from functools import partial
from typing import Union, TypeVar, Type, Optional
from cvt_tensorflow.models.layers.utils import Dense_

L = TypeVar("L",bound=tf.keras.layers.Layer)
I = TypeVar("I",bound=tf.keras.initializers.Initializer)


@tf.keras.utils.register_keras_serializable(package='cvt')
class Mlp(tf.keras.layers.Layer):
    """
    Multi-Layer Perceptron (MLP) block
    """
    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Type[L] = partial(tfa.layers.GELU,
                                              approximate = False
                                              ),
                 drop: float = 0.,
                 dense_kernel_initializer: Union[Type[I],str,None] = None,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        in_features : int
            Input features dimension.
        hidden_features : int, optional
            Hidden features dimension. 
            The default is None.
        out_features : int, optional
            Output features dimension. 
            The default is None.
        act_layer : tf.keras.layers.Layer, optional
            Activation layer. 
            The default is tfa.layers.GELU with
            approximate set to False
        drop : int, optional
            Dropout rate. 
            The default is 0.
        dense_kernel_initializer: tf.keras.initializers.Initializer or str, 
                                  optional
            Kernel initializer for fully connected layer.  
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        self.drop = drop
        self.dense_kernel_initializer = dense_kernel_initializer
    
    def build(self,input_shape): 
        self._out_features = self.out_features or self.in_features
        self._hidden_features = self.hidden_features or self.in_features
        self.fc1 = Dense_(in_features = self.in_features,
                          out_features = self._hidden_features,
                          kernel_initializer = self.dense_kernel_initializer,
                          name = "fc1"
                          )
        self.act = self.act_layer(name = "act")
        self.fc2 = Dense_(in_features = self._hidden_features,
                          out_features = self._out_features,
                          kernel_initializer = self.dense_kernel_initializer,
                          name = "fc2"
                          )
        self._drop = tf.keras.layers.Dropout(rate = self.drop,
                                             name = "drop"
                                             )
        super().build(input_shape)

    def call(self,inputs,**kwargs):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self._drop(x)
        x = self.fc2(x)
        x = self._drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'in_features': self.in_features,
                       'hidden_features': self.hidden_features,
                       'out_features': self.out_features,
                       'act_layer': self.act_layer,
                       'drop': self.drop,
                       'dense_kernel_initializer': self.dense_kernel_initializer
                       })
        return config