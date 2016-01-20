#!/bin/bash

IFS=$'\n'

for file in $(find ../train-512x512/ -name *.jpeg | sed '1d'); do
    ./histmatch -c rgb ../train-512x512/13_left.jpeg $file $(basename $file)
done
