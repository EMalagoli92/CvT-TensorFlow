__author__ = "EMalagoli92"

import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed

# Set Seed
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
random_seed.set_seed(SEED)
np.random.seed(SEED)

with open(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "version.json")), "r"
) as handle:
    __version__ = json.load(handle)["version"]


from .models.cvt import CvT
