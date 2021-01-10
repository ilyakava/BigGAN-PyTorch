#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python sample.py \
--dataset I64_hdf5 --parallel --shuffle --batch_size 128  \
--num_workers 1 --num_G_accumulations 1 --num_D_accumulations 1 \
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
--use_multiepoch_sampler \
--sample_np_mem \
--official_IS \
--G_eval_mode \
--experiment_name mh_noconcat_4step \
--mh_csc_loss \
--model=BigGANmh \
--load_weights '040000,036000,032000,028000,024000,020000,016000,012000,008000,004000' \
--sample_multiple
#--resume
#--resampling



# --load_in_mem
#--G_ch 96 --D_ch 96 \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data/imagenet \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \