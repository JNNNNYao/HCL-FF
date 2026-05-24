# HCL-FF: Hierarchical and Contrastive Learning for Forward-Forward Algorithm [CVPR 2026]

[Jie-En Yao](https://jnnnnyao.github.io/), [Hong-En Chen](https://scholar.google.com/citations?user=orq2dLAAAAAJ&hl=en), [C.-C. Jay Kuo](https://scholar.google.com/citations?user=81d60okAAAAJ&hl=zh-TW)

This is an official PyTorch implementation of the paper **HCL-FF: Hierarchical and Contrastive Learning for Forward-Forward Algorithm [CVPR 2026]**. In this work, we:

- Identifies two fundamental limitations of Forward-Forward learning: the lack of hierarchical coordination and the decoupling dilemma that leaves goodness-decoupled features semantically unconstrained.
- Proposes HCL-FF, integrating coarse-to-fine hierarchical supervision and supervised contrastive learning to structure both supervision across depth and feature geometry after goodness decoupling.
- Achieves state-of-the-art performance among FF methods, with substantial gains on fine-grained datasets.

## Hierarchy Construction
MNIST
```bash
cd data
python3 MNIST_FMNIST_hierarchy.py --dataset mnist
```

Fashion MNIST
```bash
cd data
python3 MNIST_FMNIST_hierarchy.py --dataset fashion_mnist
```

Cifar-10
```bash
python main.py --task cifar10 --epochs 200 --train_batch_size 128 --lr_min 0.008 --device 0 --task_dir cifar10_pretrain --pretrain
cd data
python3 data_driven_hierarchy.py --dataset cifar10 --pretrain_dir cifar10_pretrain
```

Cifar-100
```bash
python main.py --task cifar100 --epochs 200 --train_batch_size 512 --lr_min 0.008 --device 0 --task_dir cifar100_pretrain --pretrain
cd data
python3 data_driven_hierarchy.py --dataset cifar100 --pretrain_dir cifar100_pretrain
```

Tiny-ImageNet
```bash
python main.py --task tinyimagenet200 --epochs 1000 --train_batch_size 512 --lr 0.04 --lr_min 0.0002 --device 0 --task_dir tinyimagenet200_pretrain --save_step 200 --pretrain
cd data
python3 data_driven_hierarchy.py --dataset tinyimagenet200 --pretrain_dir tinyimagenet200_pretrain
```

## HCL-FF Training
MNIST
```bash
python main.py --task mnist --epochs 50 --train_batch_size 128 --lr_min 0.06 --device 0 --task_dir mnist
```

Fashion MNIST
```bash
python main.py --task fmnist --epochs 150 --train_batch_size 128 --lr_min 0.008 --device 0 --task_dir fmnist
```

Cifar-10
```bash
python main.py --task cifar10 --epochs 1000 --train_batch_size 128 --lr_min 0.0002 --device 0 --task_dir cifar10
```

Cifar-100
```bash
python main.py --task cifar100 --epochs 1000 --train_batch_size 512 --lr_min 0.0002 --device 0 --task_dir cifar100
```

Tiny-ImageNet
```bash
python main.py --task tinyimagenet200 --epochs 1000 --train_batch_size 512 --lr 0.04 --lr_min 0.0002 --device 0 --task_dir tinyimagenet200 --save_step 50
```

## Acknowledgements

Our code was built on [DeeperForward](https://github.com/tobysunsun/deeperforward). We would like to express our gratitude to the authors for generously sharing their code and contributing to the community.