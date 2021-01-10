#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python sample.py \
--shuffle --batch_size 50 --G_batch_size 64 --parallel \
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
--experiment_name cifar100_baseline_redo_feb1 \
--sample_np_mem \
--official_FID \
--official_IS \
--dataset_is_fid /fs/vulcan-scratch/ilyak/locDoc/data/cifar100/fid_is_scores.npz \
--G_eval_mode \
--sample_multiple \
--load_weights '002500,005000,007500,010000,012500,015000,017500,020000,022500,025000,027500,030000,032500,035000,037500,040000,042500,045000,047500,050000,052500,055000,057500,060000,062500,065000,067500,070000,072500,075000,077500,080000,082500,085000,087500,090000,092500,095000,097500,100000'

CUDA_VISIBLE_DEVICES=0 python sample.py \
--shuffle --batch_size 50 --G_batch_size 64 --parallel \
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
--experiment_name cifar100_baseline_redo_feb2 \
--sample_np_mem \
--official_FID \
--official_IS \
--dataset_is_fid /fs/vulcan-scratch/ilyak/locDoc/data/cifar100/fid_is_scores.npz \
--G_eval_mode \
--sample_multiple \
--load_weights '002500,005000,007500,010000,012500,015000,017500,020000,022500,025000,027500,030000,032500,035000,037500,040000,042500,045000,047500,050000,052500,055000,057500,060000,062500,065000,067500,070000,072500,075000,077500,080000,082500,085000,087500,090000,092500,095000,097500,100000'


# mh saved in 2000 steps
# --load_weights '100000,098000,096000,094000,092000,090000,088000,086000,084000,082000,080000,078000,076000,074000,072000,070000,068000,066000,064000,062000,060000,058000,056000,054000,052000,050000,048000,046000,044000,042000,040000,038000,036000,034000,032000,030000,028000,026000,024000,022000,020000,018000,016000,014000,012000,010000,008000,006000,004000,002000' \


# baselines saved in 2500 steps
# --load_weights '002500,005000,007500,010000,012500,015000,017500,020000,022500,025000,027500,030000,032500,035000,037500,040000,042500,045000,047500,050000,052500,055000,057500,060000,062500,065000,067500,070000,072500,075000,077500,080000,082500,085000,087500,090000,092500,095000,097500,100000' \

# --data_root /scratch0/ilya/locDoc/data/cifar10 \
# --weights_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --logs_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --samples_root /scratch0/ilya/locDoc/BigGAN/cifartest \


