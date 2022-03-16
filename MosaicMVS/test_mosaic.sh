#!/usr/bin/env bash

TESTPATH="./scene1/"
TESTLIST="./lists/SAOI/test.txt"
CKPT_FILE="./casmvsnet.ckpt"
OUTDIR="./output_mosaic"
CUDA_VISIBLE_DEVICES=1 python test_mosaic.py --dataset=dataset_mosaic --batch_size=1 --numdepth=400 --testlist=$TESTLIST --loadckpt=$CKPT_FILE --testpath_single_scene=$TESTPATH --outdir=$OUTDIR --num_view=25