#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python sample.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ch 96 --D_ch 96 \
--G_shared \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--hier --dim_z 120 --shared_dim 128 \
--G_init ortho --D_init ortho \
--data_root /scratch0/ilya/locDoc/data/cifar10 \
--weights_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--logs_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--samples_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--experiment_name table1row7 \
--sample_np_mem \
--official_IS \
--dataset_is_fid /scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz \
--G_eval_mode \
--load_weights '003000,002900,002800,002700,002600,002500,002400,002300,002200,002100,002000,001900,001800,001700,001600,001500,001400,001300,001200,001100,001000,000900,000800,000700,000600,000500,000400,000300,000200,000100' \
--sample_multiple
#--overwrite