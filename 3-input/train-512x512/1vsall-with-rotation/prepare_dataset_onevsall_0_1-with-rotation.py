#!/usr/bin/python

import pandas as pd
import glob
import os
from optparse import OptionParser
from PIL import Image
import random


def main():
    random.seed(1234)

    test_set_fraction = 0.1
    extension = ".jpeg"
    
    desired_samples = 400000

    print("Desired training set size: " + str(desired_samples))

# Suppose that we want 600000 images as training + test set because we now that they with a resolution of
# 512x512x3 occupay approximately 13GB as compressed as JPEG (Large but enough to be in RAM)
#
# Data augmentation proposed:
# Always normal + hflip + vflip + R180
# When required: normal + hflip + vflip randomly rotated
# In classification 1 vs all we now that we have:
# Example: For
# 25810 0-class images
# 9283 1-class images (1-2-3-4 classes)
# So:
# 400000/2 = 200000 = 25810*(4 + 3*RR0) for  class 0
# 400000/2 = 200000 = 9283*(4 + 3*RR1) for class 1
#
# RR0 = 1.2496
# RR1 = 5.8483
#
#
# desired_samples/2 = zeros*(4 + 3*RR0)
# desired_samples/2 = ones*(4 + 3*RR1)
#
# RR0 = maxtimes_0 = (desired_samples/(2*zeros) - 4) / 3
# RR1 = maxtimes_1 = (desired_samples/(2*ones) - 4) / 3

    usage = "Generates a 1 vs all paired image set from a class unpaired image set"
    parser = OptionParser(usage = usage)
    parser.add_option("-f", "--file", dest="filename",
                      help="CSV file containing labels", metavar="FILE")
    parser.add_option("-o", "--ofile", dest="ofilename",
                      help="CSV output file with the new image names and labels", metavar="FILE")
    parser.add_option("-s", "--src", dest="directory_src",
                      help="Source directory containing the JPEG files (.jpeg extension)")
    parser.add_option("-d", "--dst", dest="directory_dst",
                      help="Destination directory containing the JPEG files")


    (options, args) = parser.parse_args()

    # create train ans test subdirectories inside destination directory
    dir = options.directory_dst + "/train"
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)

    dir = options.directory_dst + "/test"
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)


    # Read the labels from CSV file
    labels = pd.read_csv(options.filename)
    labels = labels.set_index(['image'])

    # Read the list of jpeg files present in directory
    file_list = glob.glob(options.directory_src + "/*" + extension)

    # Mark labels present in file_list
    labels['present'] = 'F'
    for f in file_list:
        base = os.path.basename(f)
        key = base[:-len(extension)]
        labels.loc[key, 'present'] = 'T'

    olabels = {}

    # count the number of occurrences of each level
    counts = pd.value_counts(labels[labels.present == 'T']['level'])

    # Proportion of 0 data augmentation required to have a paired dataset between 0 vs 1,2,3,4
    # We have to take into account the data augmentation of the 1,2,3,4 part => 4 times (normal, hflip, vflip and r180)
    zeros = float(counts[0])
    ones = float(counts[1] + counts[2] + counts[3] + counts[4])
    
    print("Initial Zeros: " + str(zeros))
    print("Initial Ones:  " + str(ones))

    maxtimes_0 = (0.5*desired_samples/float(zeros) - 4.0) / 3.0
    maxtimes_1 = (0.5*desired_samples/float(ones) - 4.0) / 3.0

    print("Zeros rotation multiplicator: " + str(maxtimes_0))
    print("Ones rotation multiplicator:  " + str(maxtimes_1))

    dst_train = options.directory_dst + "/train/"
    dst_test = options.directory_dst + "/test/"
    for f in file_list:
        if random.random() <= test_set_fraction:
            dst = dst_test
        else:
            dst = dst_train

        base = os.path.basename(f)
        key = base[:-len(extension)]
        level = labels.loc[key].level

        if level == 0:
            maxtimes = maxtimes_0
        else:
            maxtimes = maxtimes_1

        im = Image.open(f)
        im.save(dst + key + extension)
        olabels[key] = level
        key2 = key + '-vflip'
        im.transpose(Image.FLIP_LEFT_RIGHT).save(dst + key2 + extension)
        olabels[key2] = level
        key3 = key + '-hflip'
        im.transpose(Image.FLIP_TOP_BOTTOM).save(dst + key3 + extension)
        olabels[key3] = level
        key4 = key + '-r180'
        im.transpose(Image.ROTATE_180).save(dst + key4 + extension)
        olabels[key4] = level
        times = 0
        while times < maxtimes:
            if times < int(maxtimes) or random.random() <= (maxtimes - int(maxtimes)):
                rot = int(360.0*random.random())
                keyr = key + "-r" + str(rot)
                im.rotate(rot).save(dst + keyr + extension)
                olabels[keyr] = level
                rot = int(360.0*random.random())
                keyr = key + '-vflip' + "-r" + str(rot)
                im.transpose(Image.FLIP_LEFT_RIGHT).rotate(rot).save(dst + keyr + extension)
                olabels[keyr] = level
                rot = int(360.0*random.random())
                keyr = key + '-hflip' + "-r" + str(rot)
                im.transpose(Image.FLIP_TOP_BOTTOM).rotate(rot).save(dst + keyr + extension)
                olabels[keyr] = level
            times = times + 1
            

    # save new labels
    df = pd.DataFrame(olabels.items(), columns=['image', 'level'])
    df.to_csv(options.ofilename, index=False, columns=['image','level'])
    print("Please check that labels CSV file contains only two rows: image and level. If not, remove the other one!")

if __name__ == "__main__":
    main()


