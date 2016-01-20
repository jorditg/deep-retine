#!/bin/bash

IFS=$'\n'

dir1="../../1-preprocess/train-1-preprocess-width/good"
dir2="../../1-preprocess/train-1-preprocess-width/bad-processed/good"
dir3="../../1-preprocess/train-1-preprocess-width/bad-processed/bad-processed/good"
dir4="../../1-preprocess/train-1-preprocess-width/bad-processed/bad-processed/bad-processed-manual-good"

# empty directory
for i in $(ls $dir1/*.jpeg); do
    ln -s $i $(basename $i)
done
for i in $(ls $dir2/*.jpeg); do
    ln -s $i $(basename $i)
done
for i in $(ls $dir3/*.jpeg); do
    ln -s $i $(basename $i)
done
for i in $(ls $dir4/*.jpeg); do
    ln -s $i $(basename $i)
done
