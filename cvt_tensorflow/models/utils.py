from itertools import repeat
import collections.abc as container_abcs
from typing import Union, List, TypeVar, Type
import tensorflow as tf

L = TypeVar("L",bound=tf.keras.layers.Layer)

def _to_channel_last(x: tf.Tensor) -> tf.Tensor:
    """
    Parameters
    ----------
    x : tf.Tensor
        Tensor of shape: (B, C, H, W)

    Returns
    -------
    tf.Tensor of shape: (B, H, W, C)
    """
    return tf.transpose(x, perm = [0, 2, 3, 1])


def _to_channel_first(x: tf.Tensor) -> tf.Tensor:
    """
    Parameters
    ----------
    x : tf.Tensor
        Tensor of shape: (B, H, W, C)

    Returns
    -------
    tf.Tensor of shape: (B, C, H, W)
    """
    return tf.transpose(x,perm=[0, 3, 1, 2])


def split_(tensor: tf.Tensor,
           split_size_or_sections: Union[int,List[int]],
           dim: int = 0
           ) -> tf.Tensor:
    '''
    Parameters
    ----------
    tensor : tf.Tensor
        Tensor to split.
    split_size_or_sections : Union[int,list]
        Size of a single chunk or list of sizes for each chunk
    dim : int, optional
        Dimension along which to split the tensor.
        The default is 0.

    Returns
    -------
    tf.tensor
        Splitted tensor
    '''
    if isinstance(split_size_or_sections,int):
        q_r = divmod(tensor.shape[dim],split_size_or_sections)
        if q_r[1] > 0:
            split_size_or_sections = [split_size_or_sections] * q_r[0] + [q_r[1]]  
    return tf.split(value = tensor,
                    num_or_size_splits = split_size_or_sections,
                    axis = dim
                    ) 

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def run_layers_list(input_: tf.Tensor,
                    list_: List[Type[L]]
                    ) -> tf.Tensor:
    '''
    Parameters
    ----------
    input_ : tf.Tensor
        Input Tensor
    list_ : list
        List of tf.keras.layers.Layer

    Returns
    -------
    x : TYPE
        Output Tensor

    '''
    x = input_
    for layer in list_:
        x = layer(x)
    return x