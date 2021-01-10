#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python sample.1.py \
--dataset STL48 --parallel --shuffle --batch_size 128  \
--bottom_width 3 \
--augment \
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
--experiment_name mh_48px_baseline \
--get_test_error \
--G_eval_mode \
--sample_multiple \
--load_weights '036000,068000,070000,072000,074000,076000,078000,080000,082000,084000,086000,088000,090000,092000,094000,096000,098000,100000'
#--sample_multiple
#--resume \
#--resampling

#--mh_csc_loss \
#--model=BigGANmh \

# --mh_csc_loss \
# --model=BigGANmh \

# this "augment" right now is just horizontal flips, no cropping.



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
