#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python sample.py \
--dataset STL48 --parallel --shuffle --batch_size 128  \
--bottom_width 3 \
--num_workers 2 --num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 2 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 0 --D_attn 0 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared --shared_dim 128 \
--G_init ortho --D_init ortho \
--hier --dim_z 120 \
--G_eval_mode \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl10 \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl10 \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl10 \
--ema --use_ema --ema_start 20000 \
--save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--test_every 20000000 \
--historical_save_every 2000 \
--experiment_name mh_48px_fmloss \
--model BigGANmh \
--mh_loss \
--sample_np_mem \
--official_IS \
--dataset_is_fid /fs/vulcan-scratch/ilyak/locDoc/data/stl10/fid_is_scores_48.npz \
--G_eval_mode \
--sample_multiple \
--load_weights '100000,098000,096000,094000,092000,090000,020000,018000,016000,014000,012000,010000,008000,006000,004000,002000'


# --model=BigGANmh \
# --sample_np_mem \
#--official_IS \

# --load_in_mem
#--G_ch 96 --D_ch 96 \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl10 \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl10 \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl10 \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data/imagenet \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/Imagenet64 \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/Imagenet64 \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/Imagenet64 \
