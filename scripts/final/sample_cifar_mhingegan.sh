#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python sample.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggancifar \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggancifar \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggancifar \
--mh_loss \
--model=BigGANmh \
--sample_np_mem \
--official_IS \
--official_FID \
--dataset_is_fid /fs/vulcan-scratch/ilyak/locDoc/data/cifar10/fid_is_scores.npz \
--G_eval_mode \
--experiment_name cifar100_fm_and_cs_p05 \
--sample_multiple \
--load_weights '042500' \

#--overwrite


# --dataset_is_fid /scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz \
# --dataset_is_fid /fs/vulcan-scratch/ilyak/locDoc/data/cifar10/fid_is_scores.npz \

# --data_root /scratch0/ilya/locDoc/data/cifar10 \
# --weights_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --logs_root /scratch0/ilya/locDoc/BigGAN/cifartest \
# --samples_root /scratch0/ilya/locDoc/BigGAN/cifartest \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggancifar \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggancifar \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggancifar \

# --sample_np_mem \
# --official_IS \

# part 1
# --experiment_name mh_csc_loss_noconcat_gap \
# --load_weights '020000,017500,015000,012500,010000,007500,005000,002500' \

# part 2
# --experiment_name mh_csc_loss_noconcat_gap_phase2_4 \
# --load_weights '041000,080000,079000,078000,077000,076000,075000,074000,073000,072000,071000,070000,069000,068000,067000,066000,065000,064000,063000,062000,061000,060000,059000,058000,057000,056000,055000,054000,053000,052000,051000,050000,049000,048000,047000,046000,045000,044000,043000,042000,040000,039000,038000,037000,036000,035000,034000,033000,032000,031000,030000,029000,028000,027000,026000,025000,024000,023000,022500' \

