# DECEMBERT: Learning from Noisy Instructional Videos via Dense Captions and Entropy Minimization

Implementation of NAACL2021 paper: [DECEMBERT: Learning from Noisy Instructional Videos via Dense Captions and Entropy Minimization](https://www.aclweb.org/anthology/2021.naacl-main.193/) by *Zineng Tang, *Jie Lei, Mohit Bansal

## Setup
```
# Create python environment (optional)
conda create -n decembert python=3.7

# Install python dependencies
pip install -r requirements.txt
```
To speed up the training, mixed precision is recommended. 
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Running
Running pre-training command
```
bash scripts/pretrain.sh 0,1,2,3
```

## Video Features Extraction Code

The feature extraction scripts is provided in the feature_extractor folder. 

We extract our 2D-level video features with ResNet152 
Github Link: [torchvision](https://github.com/pytorch/vision)

We extract our 3D-level video features with 3D-ResNext
Github Link: [3D-RexNext](https://github.com/kenshohara/3D-ResNets-PyTorch) 

## Dense Captions Extraction Code

Following the implementation of dense captioning aided pre-training, we pre-extract dense captions with the following code. 

Original Github Link: [Dense Captioning with Joint Inference and Visual Context](https://github.com/linjieyangsc/densecap) (pytorch reproduced)

Important todos are to change the framerate sampling in code implementation according to dfferent video types.

## Dataset Links

### Pre-training Dataset

[Howto100m](https://www.di.ens.fr/willow/research/howto100m/)

### Downstream Dataset

MSRVTT

[MSRVTT-QA](https://github.com/xudejing/video-question-answering)

[Youcook2](http://youcook2.eecs.umich.edu/)


(TODO: add downstream tasks)

## Reference
```
@inproceedings{tang2021decembert,
  title={DeCEMBERT: Learning from Noisy Instructional Videos via Dense Captions and Entropy Minimization},
  author={Tang, Zineng and Lei, Jie and Bansal, Mohit},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={2415--2426},
  year={2021}
}
```

## Acknowledgement

Part of the code is built based on huggingface [transformers](https://github.com/huggingface/transformers) and facebook
[faiss](https://github.com/facebookresearch/faiss) and [TVCaption](https://github.com/jayleicn/TVCaption).
