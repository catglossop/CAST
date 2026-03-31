#! /bin/bash

# Split the atomic dataset into train and test sets
python data_split.py --data-dir <source dir> --dataset-name atomic_left --split 0.8
python data_split.py --data-dir <source dir> --dataset-name atomic_adjust_left --split 0.8
python data_split.py --data-dir <source dir> --dataset-name atomic_right --split 0.8
python data_split.py --data-dir <source dir> --dataset-name atomic_adjust_right --split 0.8
python data_split.py --data-dir <source dir> --dataset-name atomic_forward --split 0.8
python data_split.py --data-dir <source dir> --dataset-name atomic_stop --split 0.8