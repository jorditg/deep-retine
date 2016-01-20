#!/bin/bash

IFS=$'\n'

mkdir bad
mkdir good

for file in $(cat bad.txt); do
 mv $file ./bad
done

for file in $(cat good.txt); do
 mv $file ./good
done
