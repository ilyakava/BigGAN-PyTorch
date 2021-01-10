#!/bin/bash
python train.py \
--dataset TI128_hdf5 --parallel --shuffle --batch_size 128  \
--load_in_mem \
--num_workers 7 --num_G_accumulations 8 --num_D_accumulations 8 \
--num_D_steps 2 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--num_epochs 1000 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared --shared_dim 128 \
--G_init ortho --D_init ortho \
--hier --dim_z 120 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 \
--save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--test_every -1 \
--historical_save_every 1000 \
--mh_loss \
--mh_loss_weight 0.05 \
--model=BigGANmh \
--experiment_name ti128_p05_bigbatch_attn64 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data/tiny-imagenet \
--weights_root /vulcan/scratch/ilyak/experiments/TinyImagenet \
--logs_root /vulcan/scratch/ilyak/experiments/TinyImagenet \
--samples_root /vulcan/scratch/ilyak/experiments/TinyImagenet \
--resume \
# --load_weights '070000'
