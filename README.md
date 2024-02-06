# U-BDD++: Unsupervised Building Damage Detection from Satellite Imagery
Code implementation of "Learning Efficient Unsupervised Satellite Image-based Building Damage Detection" from ICDM 2023.

[[Paper on ArXiv](https://arxiv.org/abs/2312.01576)] [[BibTeX](#citation)]

## Overview
This repository contains code for U-BDD++.

## U-BDD Benchmark

### Data Preparation
Our work uses the public xBD dataset from xView2 challenge. You can find the dataset from [here](https://xview2.org/dataset) (account required). Please download the "Challenge training set", "Challenge test set" and "Challenge holdout set" datasets and follow the instructions on the website to unpack the files.

After downloading the dataset, the file structure should be similar to:
```
[xBD root folder]
├── hold
│   ├── images
│   └── labels
├── test
│   ├── images
│   └── labels
└── train
    ├── images
    └── labels
```

Firstly, the data needs to be preprocessed before training. Please run the following command to preprocess the data:
```sh
python datasets/preprocess-data.py --data_dir <path to xBD root folder>
```
This will create a new folder `masks` under each dataset split folder, which contains the damage masks for each building.


## Installation
To start, please clone this repository to your local machine and follow the instructions below.

### Requirements
This repository requires `python>=3.9`, `pytorch>=1.13` and `torchvision>=0.14`. Older versions may work, but they are not tested.

```sh
pip install git+https://github.com/facebookresearch/segment-anything.git

pip install git+https://github.com/IDEA-Research/GroundingDINO.git

pip install -r requirements.txt
```

> [!NOTE]
> As per installation requirement from Grounding DINO, please make sure the environment variable `CUDA_HOME` is set.
`export CUDA_HOME=/path/to/cuda-xx.x`
>
> Additionally, DINO requires building the custom PyTorch ops:
> ```sh
> cd models/dino/ops
> python setup.py build install
> ```

<!-- To install the requirements, please run:
```sh
pip install -r requirements.txt
``` -->

### Pre-trained Weights
You can download the pre-trained weights of U-BDD++ for evaluation.

[Coming Soon]

## Evaluation

To evaluate U-BDD++ on xBD dataset, please run:
```sh
CUDA_VISIBLE_DEVICES=0 python predict.py --test-set-path "path/to/xbd/test" --dino-path "path/to/dino/weights" --dino-config "path/to/dino/config" --sam-path "path/to/sam/weights"

# for example
CUDA_VISIBLE_DEVICES=0 python predict-pretrain.py --test-set-path "/home/datasets/xbd/test" --dino-path "/home/outputs/dino/resnet/bld-det-pl-2023-06-22-19-53-11/checkpoint0011.pth" --dino-config "/home/U-BDD/models/dino/config/DINO_4scale_UBDD_resnet.py" --sam-path "/home/checkpoints/SAM/sam_vit_h_4b8939.pth"
```


## License
This repository is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.


## Attribution
Part of this repository used the following repositories:

Related repositories:
- [DINO](https://github.com/IDEA-Research/DINO)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [CLIP](https://github.com/openai/CLIP)

- [BDANet](https://github.com/ShaneShen/BDANet-Building-Damage-Assessment)

Thanks to the authors for their great work!

## Citation
If you find this repository useful in your research, please use the following BibTeX for citation:

```bibtex
@article{zhang2023learning,
  title={Learning Efficient Unsupervised Satellite Image-based Building Damage Detection},
  author={Zhang, Yiyun and Wang, Zijian and Luo, Yadan and Yu, Xin and Huang, Zi},
  journal={arXiv preprint arXiv:2312.01576},
  year={2023}
}
```
