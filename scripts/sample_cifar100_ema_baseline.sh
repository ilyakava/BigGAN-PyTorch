#!/bin/bash
python sample.py \
--shuffle --batch_size 50 --parallel \
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
--experiment_name cifar100_baseline_redo \
--historical_save_every 2500 \
--global_average_pooling \

--dataset_is_fid /scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz \
--G_eval_mode \
--load_weights '002500,005000,007500,010000,012500,015000,017500,020000,022500,025000,027500,030000,032500,035000,037500,040000,042500,045000,047500,050000,052500,055000,057500,060000,062500,065000,067500,070000,072500,075000,077500,080000,082500,085000,087500,090000,092500,095000,097500,100000,145000' \
--sample_multiple
#--load_weights 012500 \
#--resume

# --data_root /scratch0/ilya/locDoc/data/cifar10 \
# --weights_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --logs_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --samples_root /scratch0/ilya/locDoc/BigGAN/cifartest \


# --sample_np_mem \
# --official_IS \

