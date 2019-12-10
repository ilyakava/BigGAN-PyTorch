#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python sample.py \
--shuffle --batch_size 2500 --G_batch_size 256 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C100 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--experiment_name cifar100_baseline_redo \
--sample_np_mem \
--official_FID \
--official_IS \
--dataset_is_fid /fs/vulcan-scratch/ilyak/locDoc/data/cifar100/fid_is_scores.npz \
--G_eval_mode \
--sample_multiple \
--load_weights '035000,032500,030000,027500,025000,022500,020000,017500,015000,012500,010000,007500,005000,002500'



# --load_weights '100000,097500,095000,092500,090000,087500,085000,082500,080000,077500,075000,072500,070000,067500,065000,062500,060000,057500,055000,052500,050000,047500,045000,042500,040000,037500,035000,032500,030000,027500,025000,022500,020000,017500,015000,012500,010000,005000' \


# --data_root /scratch0/ilya/locDoc/data/cifar10 \
# --weights_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --logs_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --samples_root /scratch0/ilya/locDoc/BigGAN/cifartest \


