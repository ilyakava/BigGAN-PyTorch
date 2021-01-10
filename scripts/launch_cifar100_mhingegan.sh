#!/bin/bash
python train.py \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 1000 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C100 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 5000 \
--save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--experiment_name cifar100_fm_and_cs_p01 \
--test_every -1 \
--historical_save_every 2000 \
--mh_loss \
--mh_loss_weight 0.01 \
--model=BigGANmh \
--resume
#--load_weights 012500 \
#--resume




# --data_root /scratch0/ilya/locDoc/data \
# --weights_root /scratch0/ilya/locDoc/BigGAN/cifar100 \
# --logs_root /scratch0/ilya/locDoc/BigGAN/cifar100 \
# --samples_root /scratch0/ilya/locDoc/BigGAN/cifar100 \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
