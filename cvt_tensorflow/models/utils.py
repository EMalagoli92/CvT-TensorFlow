import collections.abc as container_abcs
from itertools import repeat
from typing import List

import tensorflow as tf


def _to_channel_last(x: tf.Tensor) -> tf.Tensor:
    """
    Parameters
    ----------
    x : tf.Tensor
        Tensor of shape: (B, C, H, W).

    Returns
    -------
    tf.Tensor
        Tensor of shape: (B, H, W, C).
    """
    return tf.transpose(x, perm=[0, 2, 3, 1])


def _to_channel_first(x: tf.Tensor) -> tf.Tensor:
    """
    Parameters
    ----------
    x : tf.Tensor
        Tensor of shape: (B, H, W, C).

    Returns
    -------
    tf.Tensor
        Tensor of shape: (B, C, H, W).
    """
    return tf.transpose(x, perm=[0, 3, 1, 2])


def run_layers_list(input_: tf.Tensor, list_: List[tf.keras.layers.Layer]) -> tf.Tensor:
    """
    Parameters
    ----------
    input_ : tf.Tensor
        Input Tensor.
    list_ : List[tf.keras.layers.Layer]
        List of tf.keras.layers.Layer.

    Returns
    -------
    tf.Tensor
        Output Tensor.
    """
    x = input_
    for layer in list_:
        x = layer(x)
    return x


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
