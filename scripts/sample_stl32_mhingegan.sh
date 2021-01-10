#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python sample.py \
--shuffle --batch_size 256 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 50000000000000 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset STL32 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 400 \
--save_every 500 --num_best_copies 1 --num_save_copies 2 --seed 0 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/stl \
--test_every 20000000 \
--historical_save_every 2500 \
--mh_csc_loss \
--model=BigGANmh \
--global_average_pooling \
--use_unlabeled_data \
--dataset_is_fid /fs/vulcan-scratch/ilyak/locDoc/data/stl10/fid_is_scores_32.npz \
--sample_np_mem \
--official_FID \
--experiment_name mhs_p05_32px \
--sample_multiple \
--load_weights '100000'


# part 1
# --experiment_name stl32_with_unlab \
# --load_weights '022500,020000,017500,015000,012500,010000,007500,005000,002500' \

# part 2
# --experiment_name stl32_with_unlab_improve_1step \
# --load_weights '100000,097500,095000,092500,090000,087500,085000,082500,080000,077500,075000,072500,070000,067500,065000,062500,060000,057500,055000,052500,050000,047500,045000,042500,040000,037500,035000,032500,030000,027500,025000' \


# --data_root /scratch0/ilya/locDoc/data \
# --weights_root /scratch0/ilya/locDoc/BigGAN/stl \
# --logs_root /scratch0/ilya/locDoc/BigGAN/stl \
# --samples_root /scratch0/ilya/locDoc/BigGAN/stl \
