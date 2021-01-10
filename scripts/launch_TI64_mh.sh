#!/bin/bash
python train.py \
--dataset TI64_hdf5 --parallel --shuffle --batch_size 512  \
--num_workers 6 --num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 2 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 0 --D_attn 0 \
--num_epochs 500 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared --shared_dim 128 \
--G_init ortho --D_init ortho \
--hier --dim_z 120 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 \
--save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data/tiny-imagenet \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet_100k \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet_100k \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet_100k \
--test_every 100 \
--historical_save_every 2500 \
--experiment_name mhinge_p1_2step_512bs \
--mh_loss \
--mh_loss_weight 0.05 \
--model=BigGANmh \
--load_in_mem \
--resume \
--experiment_name mhinge_p1_2step_ctd1 \
--load_weights 077500

#--num_D_steps 2 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \


# --data_root /fs/vulcan-scratch/ilyak/locDoc/data/tiny-imagenet \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/tiny-half-imagenet \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/tiny-half-imagenet \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/tiny-half-imagenet \

#--resume
#--resampling

# 256@64 fit in 1 gpu of 25 GB

# 64@128 fit on 1 gpu

# --load_in_mem
#--G_ch 96 --D_ch 96 \

# --data_root /scratch0/ilya/locDoc/data/tiny-imagenet \
# --weights_root /scratch0/ilya/locDoc/BigGAN/TinyImagenet64 \
# --logs_root /scratch0/ilya/locDoc/BigGAN/TinyImagenet64 \
# --samples_root /scratch0/ilya/locDoc/BigGAN/TinyImagenet64 \
