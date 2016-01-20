#!/bin/bash

DN="train-1-preprocess-width"
IFS=$'\n'

for file in $(cat $DN-bad.txt); do
    mv ./$DN/$file ./$DN/bad
done

for file in $(cat $DN-good.txt); do
    mv ./$DN/$file ./$DN/good
done
