#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python sample.py \
--shuffle --batch_size 50 --G_batch_size 64 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /scratch2/ilyak/locDoc/data \
--weights_root /scratch2/ilyak/locDoc/BigGAN/cifar10 \
--logs_root /scratch2/ilyak/locDoc/BigGAN/cifar10 \
--samples_root /scratch2/ilyak/locDoc/BigGAN/cifar10 \
--experiment_name baseline_redo_feb5 \
--sample_np_mem \
--official_IS \
--official_FID \
--dataset_is_fid /scratch2/ilyak/locDoc/data/cifar10/fid_is_scores.npz \
--G_eval_mode \
--sample_multiple \
--load_weights '002500,005000,007500,010000,012500,015000,017500,020000,022500,025000,027500,030000,032500,035000,037500,040000,042500,045000,047500,050000,052500,055000,057500,060000,062500,065000,067500,070000,072500,075000,077500,080000,082500,085000,087500,090000,092500,095000,097500,100000'


CUDA_VISIBLE_DEVICES=1 python sample.py \
--shuffle --batch_size 50 --G_batch_size 64 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /scratch2/ilyak/locDoc/data \
--weights_root /scratch2/ilyak/locDoc/BigGAN/cifar10 \
--logs_root /scratch2/ilyak/locDoc/BigGAN/cifar10 \
--samples_root /scratch2/ilyak/locDoc/BigGAN/cifar10 \
--experiment_name mhs_p05_redofeb5 \
--mh_loss \
--model=BigGANmh \
--sample_np_mem \
--official_IS \
--official_FID \
--dataset_is_fid /scratch2/ilyak/locDoc/data/cifar10/fid_is_scores.npz \
--G_eval_mode \
--sample_multiple \
--load_weights '100000,098000,096000,094000,092000,090000,088000,086000,084000,082000,080000,078000,076000,074000,072000,070000,068000,066000,064000,062000,060000,058000,056000,054000,052000,050000,048000,046000,044000,042000,040000,038000,036000,034000,032000,030000,028000,026000,024000,022000,020000,018000,016000,014000,012000,010000,008000,006000,004000,002000'

