#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--save_every 1000000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /scratch2/ilyak/locDoc/data \
--weights_root /scratch2/ilyak/locDoc/BigGAN/cifar10 \
--logs_root /scratch2/ilyak/locDoc/BigGAN/cifar10 \
--samples_root /scratch2/ilyak/locDoc/BigGAN/cifar10 \
--test_every -1 \
--historical_save_every 2500 \
--experiment_name baseline_redo_feb5
# --load_weights 087500 \
# --resume