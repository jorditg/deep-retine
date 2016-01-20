#!/bin/bash

IFS=$'\n'

SOURCE_FILES="../train/*.jpeg"
SIZE="256x256"
FILTER="Lanczos"

for file in $(ls $SOURCE_FILES); do
    convert $file -filter $FILTER -resize $SIZE -background black -gravity center -extent $SIZE $(basename $file)
done

