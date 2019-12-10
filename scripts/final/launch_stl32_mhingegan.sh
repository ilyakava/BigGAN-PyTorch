#!/bin/bash
python train.py \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 50000000000000 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--augment \
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
--test_every -1 \
--experiment_name mhs_p05_32px \
--historical_save_every 2500 \
--global_average_pooling \
--mh_loss \
--mh_loss_weight 0.05 \
--model=BigGANmh \
--use_unlabeled_data


# --data_root /fs/vulcan-scratch/ilyak/locDoc/data \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl \