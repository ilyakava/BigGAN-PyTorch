#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python sample.py \
--shuffle --batch_size 50 --G_batch_size 64 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /scratch0/ilya/locDoc/data/cifar10 \
--weights_root /scratch0/ilya/locDoc/BigGAN/cifar10 \
--logs_root /scratch0/ilya/locDoc/BigGAN/cifar10 \
--samples_root /scratch0/ilya/locDoc/BigGAN/cifar10 \
--experiment_name baseline_redo_feb3 \
--sample_sheets \
--dataset_is_fid /scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz \
--G_eval_mode \
--sample_multiple \
--load_weights '100000'
# --load_weights '002500,005000,007500,010000,012500,015000,017500,020000,022500,025000,027500,030000,032500,035000,037500,040000,042500,045000,047500,050000,052500,055000,057500,060000,062500,065000,067500,070000,072500,075000,077500,080000,082500,085000,087500,090000,092500,095000,097500,100000'
# --sample_np_mem \
# --official_IS \
# --official_FID \

# CUDA_VISIBLE_DEVICES=1 python sample.py \
# --shuffle --batch_size 50 --G_batch_size 64 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --dataset C10 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --data_root /scratch0/ilya/locDoc/data/cifar10 \
# --weights_root /scratch0/ilya/locDoc/BigGAN/cifar10 \
# --logs_root /scratch0/ilya/locDoc/BigGAN/cifar10 \
# --samples_root /scratch0/ilya/locDoc/BigGAN/cifar10 \
# --experiment_name baseline_redo_feb4 \
# --sample_np_mem \
# --official_IS \
# --official_FID \
# --dataset_is_fid /scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz \
# --G_eval_mode \
# --sample_multiple \
# --load_weights '002500,005000,007500,010000,012500,015000,017500,020000,022500,025000,027500,030000,032500,035000,037500,040000,042500,045000,047500,050000,052500,055000,057500,060000,062500,065000,067500,070000,072500,075000,077500,080000,082500,085000,087500,090000,092500,095000,097500,100000'

