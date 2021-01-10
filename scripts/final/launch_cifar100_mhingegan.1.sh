#!/bin/bash
python train.py \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
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
--experiment_name c100_mh_p05_redo_feb2_ctd_c \
--test_every 500 \
--historical_save_every 500 \
--mh_loss \
--model=BigGANmh \
--resume \
--mh_loss_weight 0.025 
# --load_weights 034000 \
#--resume




# --data_root /scratch0/ilya/locDoc/data \
# --weights_root /scratch0/ilya/locDoc/BigGAN/cifar100 \
# --logs_root /scratch0/ilya/locDoc/BigGAN/cifar100 \
# --samples_root /scratch0/ilya/locDoc/BigGAN/cifar100 \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
