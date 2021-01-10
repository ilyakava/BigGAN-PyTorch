#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python sample.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /scratch0/ilya/locDoc/data/cifar10 \
--weights_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--logs_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--samples_root /scratch0/ilya/locDoc/BigGAN/cifartest \
--model=BigGANmh \
--sample_np_mem \
--official_IS \
--dataset_is_fid /scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz \
--G_eval_mode \
--experiment_name mh_csc_loss_noconcat_gap_phase2_4 \
--load_weights '045000,046000,047000,048000,049000,050000,051000,052000,053000,054000,055000,056000,057000,058000,059000,060000,061000,062000,063000,064000,065000,066000,067000,068000,069000,070000,071000,072000,073000,074000,075000,076000,077000,078000,079000,080000,081000,082000,083000,084000,085000,086000,087000,088000,089000,090000,091000,092000,093000,094000' \
--sample_multiple