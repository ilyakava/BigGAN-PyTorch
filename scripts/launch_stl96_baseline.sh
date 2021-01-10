#!/bin/bash
python3 train.py \
--dataset STL96 --parallel --shuffle --batch_size 128  \
--bottom_width 3 \
--num_workers 6 --num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 2 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 0 --D_attn 0 \
--augment \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared --shared_dim 128 \
--G_init ortho --D_init ortho \
--hier --dim_z 120 \
--G_eval_mode \
--data_root /scratch2/ilyak/locDoc/data \
--weights_root /scratch2/ilyak/locDoc/BigGAN/stl \
--logs_root /scratch2/ilyak/locDoc/BigGAN/stl \
--samples_root /scratch2/ilyak/locDoc/BigGAN/stl \
--ema --use_ema --ema_start 40000 \
--save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler \
--test_every 20000000 \
--historical_save_every 2000 \
--experiment_name baseline96_noattn \
--ignore_projection_discriminator \
--use_unlabeled_data \
--num_epochs 5000 \
--resume \
--load_weights 018000
#--resampling

# this "augment" right now is just horizontal flips, no cropping.



# --load_in_mem
#--G_ch 96 --D_ch 96 \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl10 \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl10 \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl10 \

# --data_root /scratch0/ilya/locDoc/data \
# --weights_root /scratch0/ilya/locDoc/BigGAN/stl \
# --logs_root /scratch0/ilya/locDoc/BigGAN/stl \
# --samples_root /scratch0/ilya/locDoc/BigGAN/stl \
