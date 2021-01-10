#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train.py \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--save_every 250 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /scratch0/ilya/locDoc/data/cifar10 \
--weights_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--logs_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--samples_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--test_every 20000000 \
--experiment_name mh_csc_loss_concat_weston_avg \
--historical_save_every 2500 \
--mh_loss \
--model=BigGANmh \
--class_concat \
--resume
