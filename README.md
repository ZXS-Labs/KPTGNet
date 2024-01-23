

# KPTGNet

![Overview](https://github.com/ZXS-Labs/KPTGNet/blob/master/doc/teaser.png)

## Requirement

- Pyhthon 3.8
- Pytorch 1.10

## Installation

Get the code.

```bash
git clone https://github.com/ZXS-Labs/KPTGNet.git
cd KPTGNet
```

Install packages with a setup file.

```bash
bash install.sh
```

Install graspnetAPI for evaluation.

```
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

## Dataset Preparation

Download the original [GraspNet ](https://graspnet.net/) dataset.

Tolerance labels are not included in the original dataset, and need additional generation. generate tolerance label by running the script:

```bash
cd dataset
sh command_generate_tolerance_label.sh
```

## Training and Testing

Training examples are shown in [command_train.sh](https://github.com/ZXS-Labs/KPTGNet/blob/master/command_train.sh). `--dataset_root`, `--camera` and `--log_dir` should be specified according to your settings. You can use TensorBoard to visualize training process.

Testing examples are shown in [command_test.sh](https://github.com/ZXS-Labs/KPTGNet/blob/master/command_test.sh), which contains inference and result evaluation. `--dataset_root`, `--camera`, `--checkpoint_path` and `--dump_dir` should be specified according to your settings. Set `--collision_thresh` to -1 for fast inference.

The pretrained weights can be downloaded from:

- `checkpoint.tar` [[Google Drive](https://drive.google.com/file/d/1nj3Fp4KEAJhMJZPPx2FnD1-4Zjn0acLO/view?usp=sharing)]

`checkpoint.tar` is trained using RealSense data.

## Acknowledgement

We thank the authors that shared the code of their works. In particular:

- Hao Shu Fang for providing the code of [graspnet-baseline](https://github.com/graspnet/graspnet-baseline?tab=readme-ov-file)
- Jinyoung Park for providing the code of [SPoTr](https://github.com/mlvlab/SPoTr)

Our work is inspired by these work.
