#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python sample.py \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 50000000000000 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset STL32 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 400 \
--save_every 500 --num_best_copies 1 --num_save_copies 2 --seed 0 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl \
--test_every 20000000 \
--experiment_name stl32_baseline_prjdisc \
--historical_save_every 2500 \
--global_average_pooling \
--get_test_error \
--G_batch_size 256 \
--use_unlabeled_data \
--ignore_projection_discriminator \
--dataset_is_fid /scratch0/ilya/locDoc/data/stl10/fid_is_32_scores.npz \
--G_eval_mode \
--sample_multiple \
--load_weights '090000,087500,085000,082500,080000,077500,075000,072500,070000,067500,065000,062500,060000,057500,055000,052500,050000,047500,045000,042500,040000,037500,035000,032500,030000,027500,025000,022500,020000,017500,015000,012500,010000,007500,005000,002500' 


# --use_unlabeled_data \
# --resume \
# --sample_np_mem \
# --official_FID \




