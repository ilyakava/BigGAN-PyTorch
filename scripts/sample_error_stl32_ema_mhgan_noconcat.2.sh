#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python sample.1.py \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 50000000000000 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--augment \
--dataset STL32 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 400 \
--save_every 500 --num_best_copies 1 --num_save_copies 2 --seed 0 \
--data_root /scratch0/ilya/locDoc/data \
--weights_root /scratch0/ilya/locDoc/BigGAN/stl \
--logs_root /scratch0/ilya/locDoc/BigGAN/stl \
--samples_root /scratch0/ilya/locDoc/BigGAN/stl \
--test_every 20000000 \
--experiment_name stl32_with_unlab \
--historical_save_every 2500 \
--mh_csc_loss \
--model=BigGANmh \
--global_average_pooling \
--get_test_error \
--mh_csc_loss \
--G_eval_mode \
--sample_multiple \
--load_weights '002500,005000,007500,010000,012500,015000,017500,020000,022500'



