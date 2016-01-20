#!/bin/bash

IFS=$'\n'

#for file in $(ls *.jpeg); do ./RGB-mean-one-file.py $file; done 

cat $1 | awk '
  BEGIN {FS = ",";}; 
  { sum1 += $2; sum2 += $3; sum3 += $4; n++;};
  END { print n","sum1/n","sum2/n","sum3/n;}
'
