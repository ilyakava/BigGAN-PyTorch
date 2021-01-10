#!/bin/bash
python train.py \
--dataset I64_hdf5 --parallel --shuffle --batch_size 256  \
--num_workers 6 --num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared --shared_dim 128 \
--G_init ortho --D_init ortho \
--hier --dim_z 120 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 \
--save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data/imagenet \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/Imagenet64 \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/Imagenet64 \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/Imagenet64 \
--test_every 20000000 \
--historical_save_every 2000 \
--experiment_name mh_noconcat_4step \
--mh_csc_loss \
--model=BigGANmh \
--load_in_mem
#--resume
#--resampling



# --load_in_mem
#--G_ch 96 --D_ch 96 \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data/imagenet \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \