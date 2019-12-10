# Code for a Multi-Hinge Loss with Conditional GANs

This is the implementation for the multi-hinge loss of [this paper](https://arxiv.org/abs/1912.04216). Forked from: https://github.com/ajbrock/BigGAN-PyTorch

## Summary

CIFAR10 best IS & FID are 9.58 & 6.40, CIFAR100 best IS & FID are 14.36 & 13.32, and STL10 best IS & FID are 12.16 & 17.44.

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
See [here](https://arxiv.org/abs/1912.04216) for exmplanations of these metrics.

Links to training code coming 12/11/19...

## Attribution

Forked from: https://github.com/ajbrock/BigGAN-PyTorch
Official IS code copied from: https://github.com/openai/improved-gan
Official FID code copied from: https://github.com/bioinf-jku/TTUR
Classification densenet copied from: https://github.com/kuangliu/pytorch-cifar
Classification wideresnet copied from: https://github.com/YU1ut/MixMatch-pytorch
