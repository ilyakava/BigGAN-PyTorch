#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python sample.py \
--dataset TIH64_hdf5 --parallel --shuffle --batch_size 128  \
--num_workers 6 --num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 2 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 0 --D_attn 0 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared --shared_dim 128 \
--G_init ortho --D_init ortho \
--hier --dim_z 120 \
--G_eval_mode \
--ema --use_ema --ema_start 20000 \
--save_every 500 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /fs/vulcan-scratch/ilyak/locDoc/data/tiny-imagenet \
--weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet_100k \
--logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet_100k \
--samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet_100k \
--mh_loss \
--model=BigGANmh \
--G_eval_mode \
--sample_np_mem \
--official_IS \
--sample_multiple \
--experiment_name tih_mhinge_p05_2step \
--load_weights '045000'

# sh scripts/sample_TIH64_mh.sh

# --experiment_name mhinge_p1_2step \
# --load_weights '002500,005000,007500,010000,012500,015000,017500,020000,022500,025000,027500,030000,032500,035000,037500,040000,042500,045000,047500,050000,052500,055000,057500,060000,062500,065000,067500,070000,072500,075000,077500,080000,082500'

# --experiment_name mhinge_p1 \
# --load_weights '002500,005000,007500,010000,012500,015000,017500,020000,022500,025000,027500,030000,032500,035000,037500,040000,042500,045000,047500,050000,052500,055000,057500'



# --sample_np_mem \
# --official_IS \

#--use_multiepoch_sampler \

# --sample_np_mem \
# --official_IS \


# --data_root /scratch0/ilya/locDoc/data/tiny-imagenet/ \
# --weights_root /scratch0/ilya/locDoc/BigGAN/TinyImagenet64 \
# --logs_root /scratch0/ilya/locDoc/BigGAN/TinyImagenet64 \
# --samples_root /scratch0/ilya/locDoc/BigGAN/TinyImagenet64 \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data/tiny-imagenet \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet_100k \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet_100k \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet_100k \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data/tiny-imagenet \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet64 \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet64 \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/TinyImagenet64 \

#--dataset_is_fid /scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz \
# --load_in_mem
#--G_ch 96 --D_ch 96 \

# --data_root /fs/vulcan-scratch/ilyak/locDoc/data/imagenet \
# --weights_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \
# --logs_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \
# --samples_root /fs/vulcan-scratch/ilyak/locDoc/experiments/biggantest4gpu \
