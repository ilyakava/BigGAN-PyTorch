#!/bin/bash
#python make_hdf5.py --dataset I128 --batch_size 256
#python calculate_inception_moments.py --dataset I128_hdf5 --data_root data

# cifar
#python make_hdf5.py --dataset C10 --batch_size 256 --data_root=/scratch0/ilya/locDoc/data/cifar10 --write_dir=/scratch0/ilya/locDoc/data/cifar10

# smaller imagenet
# curl http://cs231n.stanford.edu/tiny-imagenet-200.zip -O
# unzip tiny-imagenet-200.zip
#python make_hdf5.py --dataset I64 --batch_size 256 --num_workers 15
#python make_hdf5.py --dataset TI64 --batch_size 256 --num_workers 15 --data_root=/scratch0/ilyak/tiny-imagenet-200 --write_dir=/fs/vulcan-scratch/ilyak/locDoc/data/tiny-imagenet

# places205
python make_hdf5.py --dataset P128 --batch_size 256 --num_workers 40 --data_root=/fs/vulcan-scratch/ilyak/locDoc/data/places205 --write_dir=/fs/vulcan-scratch/ilyak/locDoc/data/places205 --write_name=PLACES