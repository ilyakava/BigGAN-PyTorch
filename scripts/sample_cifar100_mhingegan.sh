#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python sample.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C100 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--test_every 20000000 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--experiment_name cifar100_fm_and_cs_p01 \
--historical_save_every 2500 \
--get_generator_error \
--model=BigGANmh \
--mh_loss \
--dataset_is_fid /fs/vulcan-scratch/ilyak/locDoc/data/cifar100/fid_is_scores.npz \
--G_eval_mode \
--sample_multiple \
--load_weights '196000,192000,188000,184000,180000,176000,172000,168000,164000,160000,156000,152000,148000,144000,140000,136000,132000,128000,124000,120000,116000,112000,108000,104000,100000'
# two


#--load_weights '196000,192000,188000,184000,180000,176000,172000,168000,164000,160000,156000,152000,148000,144000,140000,136000,132000,128000,124000,120000,116000,112000,108000,104000,100000'
# two

#--load_weights '198000,194000,190000,186000,182000,178000,174000,170000,166000,162000,158000,154000,150000,146000,142000,138000,134000,130000,126000,122000,118000,114000,110000,106000,102000'
# one


# --dataset_is_fid /fs/vulcan-scratch/ilyak/locDoc/data/cifar100/fid_is_scores.npz \


# --data_root /fs/vulcan-scratch/ilyak/locDoc/data \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \


# --data_root /scratch0/ilya/locDoc/data/cifar10 \
# --weights_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --logs_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --samples_root /scratch0/ilya/locDoc/BigGAN/cifartest \

# --sample_np_mem \
# --official_IS \
# --official_FID \

#--get_test_error \

# --load_weights '010000,012500,015000,017500,020000,022500,025000,027500,030000,032500,035000,037500,040000' 
