#!/bin/bash

IFS=$'\n'

for file in $(cat preprocess-cut-width-bad-good.txt); do
    mv $file ./good
done
