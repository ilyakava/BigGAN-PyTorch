#!/bin/bash
python train.py \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data/cifar10 \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggancifar \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggancifar \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggancifar \
--test_every 20000000 \
--experiment_name fmloss \
--historical_save_every 2500 \
--fm_loss \
--mh_csc_loss \
--model=BigGANmh \
--resampling



# --data_root /scratch0/ilya/locDoc/data/cifar10 \
# --weights_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --logs_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --samples_root /scratch0/ilya/locDoc/BigGAN/cifartest \