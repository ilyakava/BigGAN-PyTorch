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
--save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--test_every 20000000 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
--experiment_name c100_mh_p05_redo_feb2 \
--historical_save_every 2500 \
--mh_loss \
--model=BigGANmh \
--sample_np_mem \
--official_IS \
--official_FID \
--dataset_is_fid /fs/vulcan-scratch/ilyak/locDoc/data/cifar100/fid_is_scores.npz \
--G_eval_mode \
--sample_multiple \
--load_weights '102000'


# CUDA_VISIBLE_DEVICES=2 python sample.py \
# --shuffle --batch_size 50 --G_batch_size 64 --parallel \
# --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
# --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
# --dataset C100 \
# --G_ortho 0.0 \
# --G_attn 0 --D_attn 0 \
# --G_init N02 --D_init N02 \
# --ema --use_ema --ema_start 1000 \
# --save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
# --test_every 20000000 \
# --data_root /fs/vulcan-scratch/ilyak/locDoc/data \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/cifar100 \
# --experiment_name c100_mh_p05_redo_feb3 \
# --historical_save_every 2500 \
# --mh_loss \
# --model=BigGANmh \
# --sample_np_mem \
# --official_IS \
# --official_FID \
# --dataset_is_fid /fs/vulcan-scratch/ilyak/locDoc/data/cifar100/fid_is_scores.npz \
# --G_eval_mode \
# --sample_multiple \
# --load_weights '100000,098000,096000,094000,092000,090000,088000,086000,084000,082000,080000,078000,076000,074000,072000,070000,068000,066000,064000,062000,060000,058000,056000,054000,052000,050000,048000,046000,044000,042000,040000,038000,036000,034000,032000,030000,028000,026000,024000,022000,020000,018000,016000,014000,012000,010000,008000,006000,004000,002000'
