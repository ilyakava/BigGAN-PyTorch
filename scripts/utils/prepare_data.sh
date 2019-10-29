#!/bin/bash
#python make_hdf5.py --dataset I128 --batch_size 256
#python calculate_inception_moments.py --dataset I128_hdf5 --data_root data

# cifar
python make_hdf5.py --dataset C10 --batch_size 256 --data_root=/scratch0/ilya/locDoc/data/cifar10 --write_dir=/scratch0/ilya/locDoc/data/cifar10

