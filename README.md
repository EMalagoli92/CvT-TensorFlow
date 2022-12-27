<div align="center">

  <a href="https://www.tensorflow.org">![TensorFLow](https://img.shields.io/badge/TensorFlow-2.X-orange?style=for-the-badge) 
  <a href="https://github.com/EMalagoli92/CvT-TensorFlow/blob/main/LICENSE">![License](https://img.shields.io/github/license/EMalagoli92/CvT-TensorFlow?style=for-the-badge) 
  <a href="https://www.python.org">![Python](https://img.shields.io/badge/python-%3E%3D%203.9-blue?style=for-the-badge)</a>  
  
</div>

# CvT-TensorFlow
TensorFlow 2.X reimplementation of [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808), Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang.
- Exact TensorFlow reimplementation of official PyTorch repo, including `timm` modules used by authors, preserving models and layers structure.
- ImageNet pretrained weights ported from PyTorch official implementation.

## Table of contents
- [Abstract](#abstract)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgement](#acknowledgement)
- [Citations](#citations)
- [License](#license)

<div id="abstract"/>

## Abstract
Convolutional vision Transformers (CvT), improves Vision Transformers (ViT) in 
performance and efficienty by introducing convolutions into ViT to yield the 
best of both designs. This is accomplished through two primary modifications: 
a hierarchy of Transformers containing a new convolutional token embedding, 
and a convolutional Transformer block leveraging a convolutional projection. 
These changes introduce desirable properties of convolutional neural networks 
(CNNs) to the ViT architecture (e.g. shift, scale, and distortion invariance) 
while maintaining the merits of Transformers (e.g. dynamic attention, 
global context, and better generalization). 
Moreover the achieved results show that the positional encoding, 
a crucial component in existing Vision Transformers, can be safely removed 
in the model, simplifying the design for higher resolution vision tasks.


![Alt text](https://raw.githubusercontent.com/EMalagoli92/CvT-TensorFlow/266afd1057827d10f0dfb842f8ef73f5b19e471d/assets/images/pipeline.svg)
<p align = "center"><sub>The pipeline of the CvT architecture. (a) Overall architecture, showing the hierarchical multi-stage
structure facilitated by the Convolutional Token Embedding layer. (b) Details of the Convolutional Transformer Block,
which contains the convolution projection as the first layer.</sub></p>

<div id="results"/>

## Results
TensorFlow implementation and ImageNet ported weights have been compared to the official PyTorch implementation on [ImageNet-V2](https://www.tensorflow.org/datasets/catalog/imagenet_v2) test set.

### Models pre-trained on ImageNet-1K
| Configuration  | Resolution | Top-1 (Original) | Top-1 (Ported) | Top-5 (Original) | Top-5 (Ported) | #Params
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| CvT-13 | 224x224 | 69.81 | 69.81 | 89.13 | 89.13 | 20M |
| CvT-13 | 384x384 | 71.31 | 71.31 | 89.97 | 89.97 | 20M |
| CvT-21 | 224x224 | 71.18 | 71.17 | 89.31 | 89.31 | 32M |
| CvT-21 | 384x384 | 71.61 | 71.61 | 89.71 | 89.71 | 32M |


### Models pre-trained on ImageNet-22K
| Configuration  | Resoluton | Top-1 (Original) | Top-1 (Ported) | Top-5 (Original) | Top-5 (Ported) | #Params
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| CvT-13 | 384x284 | 71.76 | 71.76 | 91.39 | 91.39 | 20M |
| CvT-21 | 384x384 | 74.97 | 74.97 | 92.63 | 92.63 | 32M |
| CvT-W24 | 384x384 | 78.15 | 78.15 | 94.48 | 94.48 | 277M | 

Max metrics difference: `9e-5`.

<div id="installation"/>

## Installation
- Install from PyPI
```
pip install cvt-tensorflow
```
- Install from Github
```
pip install git+https://github.com/EMalagoli92/CvT-TensorFlow
```
- Clone the repo and install necessary packages 
```
git clone https://github.com/EMalagoli92/CvT-TensorFlow.git
pip install -r requirements.txt
```

Tested on *Ubuntu 20.04.4 LTS x86_64*, *python 3.9.7*.

<div id="usage"/>

## Usage
- Define a custom CvT configuration.
```python
from cvt_tensorflow import CvT

# Define a custom CvT configuration
model = CvT(
    in_chans=3,
    num_classes=1000,
    classifier_activation="softmax",
    data_format="channels_last",
    spec={
        "INIT": "trunc_norm",
        "NUM_STAGES": 3,
        "PATCH_SIZE": [7, 3, 3],
        "PATCH_STRIDE": [4, 2, 2],
        "PATCH_PADDING": [2, 1, 1],
        "DIM_EMBED": [64, 192, 384],
        "NUM_HEADS": [1, 3, 6],
        "DEPTH": [1, 2, 10],
        "MLP_RATIO": [4.0, 4.0, 4.0],
        "ATTN_DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_PATH_RATE": [0.0, 0.0, 0.1],
        "QKV_BIAS": [True, True, True],
        "CLS_TOKEN": [False, False, True],
        "QKV_PROJ_METHOD": ["dw_bn", "dw_bn", "dw_bn"],
        "KERNEL_QKV": [3, 3, 3],
        "PADDING_KV": [1, 1, 1],
        "STRIDE_KV": [2, 2, 2],
        "PADDING_Q": [1, 1, 1],
        "STRIDE_Q": [1, 1, 1],
    },
)
```
- Use a predefined CvT configuration.
```python
from cvt_tensorflow import CvT

model = CvT(
    configuration="cvt-21", data_format="channels_last", classifier_activation="softmax"
)
model.build((None, 224, 224, 3))
print(model.summary())
```
```
Model: "cvt-21"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 stage0 (VisionTransformer)  multiple                  62080     
                                                                 
 stage1 (VisionTransformer)  multiple                  1920576   
                                                                 
 stage2 (VisionTransformer)  ((None, 384, 14, 14),     29296128  
                              (None, 1, 384))                    
                                                                 
 norm (LayerNorm_)           (None, 1, 384)            768       
                                                                 
 head (Linear_)              (None, 1000)              385000    
                                                                 
 pred (Activation)           (None, 1000)              0         
                                                                 
=================================================================
Total params: 31,664,552
Trainable params: 31,622,696
Non-trainable params: 41,856
_________________________________________________________________
```
- Train from scratch the model.
```python    
# Example
model.compile(
    optimizer="sgd",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", "sparse_top_k_categorical_accuracy"],
)
model.fit(x, y)
```
- Use ported ImageNet pretrained weights
```python
# Example
from cvt_tensorflow import CvT

# Use cvt-13-384x384_22k ImageNet pretrained weights
model = CvT(
    configuration="cvt-13",
    pretrained=True,
    pretrained_resolution=384,
    pretrained_version="22k",
    classifier_activation="softmax",
)
y_pred = model(image)
```

<div id="acknowledgement"/>

## Acknowledgement
[CvT](https://github.com/microsoft/CvT) (Official PyTorch implementation)


<div id="citations"/>

## Citations
```bibtex
@article{wu2021cvt,
  title={Cvt: Introducing convolutions to vision transformers},
  author={Wu, Haiping and Xiao, Bin and Codella, Noel and Liu, Mengchen and Dai, Xiyang and Yuan, Lu and Zhang, Lei},
  journal={arXiv preprint arXiv:2103.15808},
  year={2021}
}
```

<div id="license"/>

## License
This work is made available under the [MIT License](https://github.com/EMalagoli92/CvT-TensorFlow/blob/main/LICENSE)