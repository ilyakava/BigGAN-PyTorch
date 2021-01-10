#!/bin/bash
python train.py \
--dataset I128_hdf5 --parallel --shuffle --batch_size 64  \
--num_workers 6 --num_G_accumulations 2 --num_D_accumulations 2 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--ema --use_ema --ema_start 20000 \
--save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data/imagenet \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/bigganhalf \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/bigganhalf \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/bigganhalf \
--test_every 20000000 \
--experiment_name half_not_mixed_default \
--G_fp16 --D_fp16

# --D_mixed_precision --G_mixed_precision 
# --load_in_mem
