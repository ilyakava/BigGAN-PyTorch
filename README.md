# Code for a Multi-Hinge Loss with K+1 GANs

This is the implementation for the multi-hinge loss of [this paper](https://openreview.net/pdf?id=kBk6w-oJ9jq) for K+1 GANs. For the main MultiHingeGAN repository please see:https://github.com/ilyakava/gan

This repository is forked from: https://github.com/ajbrock/BigGAN-PyTorch

Please cite:
```
@inproceedings{kavalerov2020study,
  title={A study of quality and diversity in K+ 1 GANs},
  author={Kavalerov, Ilya and Czaja, Wojciech and Chellappa, Rama},
  booktitle={''I Can't Believe It's Not Better!''NeurIPS 2020 workshop},
  year={2020}
}
```

## Summary

CIFAR10 best IS & FID are 9.58 & 6.40, CIFAR100 best IS & FID are 14.36 & 13.32, and STL10 best IS & FID are 12.16 & 17.44.

## Installation

### Tested on

Python 3.7.3, pytorch '1.0.1.post2', tensorflow '1.13.1', cuda 10.0.130, cudnn 7.5.0, on rhel linux 7.7 (Maipo).

Also tested on [this docker image](https://hub.docker.com/r/vastai/pytorch).

### Additional packages

`pip install scipy h5py tqdm`

## Running

### CIFAR100

Train with:

```
sh scripts/final/launch_cifar100_mhingegan.sh
```

After adjusting read dir `--data_root` and write dirs `--weights_root`, `--logs_root`, `--samples_root`.
Samples will be created automatically in `--samples_root`.

![c100_samples_49500](imgs/c100_samples_49500.jpeg)

Using the historical saves run:

```
sh scripts/final/sample_cifar100_mhingegan.sh
```

Which saves IS/FID numbers into `scoring_hist.npy` in the `--weights_root` + `--experiment_name` directory.
You can make plots like:

![c100_IS](imgs/c100_IS.png)
![c100_FID](imgs/c100_FID.png)

Get the same numbers for the BigGAN baseline via:

```
sh scripts/final/launch_cifar100_baseline.sh
sh scripts/final/sample_cifar100_baseline.sh
```

### CIFAR10

Train with:

```
sh scripts/final/launch_cifar_mhingegan.sh
```

Samples will be created automatically in `--samples_root`.

![c10_samples](imgs/c10_best_64k.jpeg)

Using the historical saves run:

```
sh scripts/final/sample_cifar_mhingegan.sh
```

![c10_IS](imgs/c10_IS.png)
![c10_FID](imgs/c10_FID.png)

Get the same numbers for the BigGAN baseline via:

```
sh scripts/final/launch_cifar_baseline.sh
sh scripts/final/sample_cifar_baseline.sh
```


### STL10 48x48

Train with:

```
sh scripts/final/launch_stl48_mhingegan.sh
```

Samples will be created automatically in `--samples_root`.

![stl48_samples](imgs/stl_best_78k.jpeg)

Using the historical saves run:

```
sh scripts/final/sample_stl48_mhingegan.sh
```

![stl48_IS](imgs/stl48_IS.png)
![stl48_FID](imgs/stl48_FID.png)

Get the same numbers for the BigGAN baseline via:

```
sh scripts/final/launch_stl48_baseline.sh
sh scripts/final/sample_stl48_baseline.sh
```

### Accuracy plots

In `scripts/final/sample*.py` change:

```
--sample_np_mem \
--official_IS \
--official_FID \
```

to `--get_train_error` or `--get_test_error` or `--get_self_error` or `--get_generator_error`.
For `--get_generator_error` adjust your paths in `sample.py` to one of the pretrained models. Or add your own.
See [here](http://bit.ly/MHingeGAN) for explanations of these metrics.

#### Training classification networks:

- [Cifar10/100](https://github.com/ilyakava/pytorch-cifar)
- [STL](https://github.com/ilyakava/MixMatch-pytorch)

## Links to pretrained models

[Google Drive](https://bit.ly/35ehl0Q)

## Attribution

Forked from: https://github.com/ajbrock/BigGAN-PyTorch
Official IS code copied from: https://github.com/openai/improved-gan
Official FID code copied from: https://github.com/bioinf-jku/TTUR
Classification densenet copied from: https://github.com/kuangliu/pytorch-cifar
Classification wideresnet copied from: https://github.com/YU1ut/MixMatch-pytorch
