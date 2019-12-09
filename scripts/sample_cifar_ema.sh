#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python sample.mh.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /scratch0/ilya/locDoc/data/cifar10 \
--weights_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--logs_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--samples_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--experiment_name mh_csc_loss_noconcat_gap_phase2_4 \
--get_test_error \
--G_eval_mode \
--mh_csc_loss \
--model BigGANmh \
--sample_multiple \
--load_weights '027000,029000,031000,033000,035000,037000,039000,041000,043000,045000,047000,049000,051000,053000,055000,057000,059000,061000,063000,065000,067000,069000,071000,073000,075000,077000,079000,081000,083000,085000,087000,089000,091000,093000'
#--overwrite
