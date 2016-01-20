#!/bin/bash

IFS=$'\n'

for file in $(cat preprocess-cut-width-bad-bad.txt); do
    mv $file ./bad
done
