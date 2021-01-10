#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python sample.2.py \
--dataset I64_hdf5 --parallel --shuffle --batch_size 64  \
--num_workers 6 --num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 2 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 0 --D_attn 0 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared --shared_dim 128 \
--G_init ortho --D_init ortho \
--hier --dim_z 120 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 \
--save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data/imagenet \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/Imagenet64 \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/Imagenet64 \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/Imagenet64 \
--experiment_name fm_and_cs_5p0 \
--mh_csc_loss \
--model=BigGANmh \
--G_eval_mode \
--sample_np_mem \
--official_IS \
--sample_multiple \
--load_weights '026000'
#--overwrite

#--use_multiepoch_sampler \

# --sample_np_mem \
# --official_IS \

# --data_root /scratch0/ilya/locDoc/data/imagenet \
# --weights_root /scratch0/ilya/locDoc/BigGAN/Imagenet64 \
# --logs_root /scratch0/ilya/locDoc/BigGAN/Imagenet64 \
# --samples_root /scratch0/ilya/locDoc/BigGAN/Imagenet64 \

#--dataset_is_fid /scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz \
# --load_in_mem
#--G_ch 96 --D_ch 96 \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data/imagenet \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \
