
## Installation

- Python (3.8.12)
- Pytorch (1.10.2+cu11.3.1)
- apex (0.1)
- [inplace-abn](https://github.com/mapillary/inplace_abn) (1.1.0)

Readily setup with the following command lines. Do remember to check your own cuda version.

```bash
# pytorch installation
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# other required packages
pip install matplotlib inplace_abn tensorboardX tensorboard termcolor

# apex installation
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Dataset

Download ADE20k and Pascal-VOC 2012 with the scripts in the `data` folder.
Feel free to create a symbolic link instead if you already have a local copy of the standard PASCAL-VOC benchmark.

**Expected dataset structure for PASCAL VOC 2012:**

    PascalVOC2012
    ├── VOCdevkit               # standard Pascal VOC benchmark
    │   ├── (VOC2007)           # optional, will not be downloaded by the script
    │   │   ├── ImageSets
    │   │   └── ...
    │   └── VOC2012
    │       ├── ImageSets
    │       └── ...
    ├── SegmentationClassAug
    ├── SegmentationClassAug_Visualization
    └── list

**Expected dataset structure for ADE20k:**

    ADEChallengeData2016          # standard ADE20k benchmark
    ├── annotations
    ├── images
    ├── objectInfo150.txt
    └── sceneCategories.txt


## Getting Started

We used the pretrained model released by the authors of [In-place ABN](https://github.com/mapillary/inplace_abn#training-on-imagenet-1k).
Create a directory named `./pretrained` and download the [weights](https://github.com/Ze-Yang/LGKD/releases/download/v1.0/resnet101_iabn_sync.pth.tar) of ResNet pretrained on ImageNet.

### Train & Evaluation with script

To reproduce our results, simply run the corresponding script (VOC 15-1 for example):

```bash
bash scripts/voc/lgkd_15-1.sh
```
