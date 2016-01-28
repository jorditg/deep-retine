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
    ones = float(4*(counts[1] + counts[2] + counts[3] + counts[4]))
    zeros_proportion = (ones - zeros) / zeros


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
            if random.random() <= zeros_proportion:
                loop = 2
            else:
                loop = 1
            indexes = [1,2,3,4]
            for i in range(1,loop+1):
                idx = random.randint(0,len(indexes)-1)
                val = indexes[idx]
                indexes.remove(indexes[idx])
                im = Image.open(f)
                key_n = key
                if val == 1:
                    im.save(dst + key + extension)  
                elif val == 2:
                    key_n = key + '-vflip'
                    im.transpose(Image.FLIP_LEFT_RIGHT).save(dst + key_n + extension)
                elif val == 3:
                    key_n = key + '-hflip'
                    im.transpose(Image.FLIP_TOP_BOTTOM).save(dst + key_n + extension)
                else:
                    key_n = key + '-r180'
                    im.transpose(Image.ROTATE_180).save(dst + key_n + extension)
                olabels[key_n] = level
        else:
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

    # save new labels
    df = pd.DataFrame(olabels.items(), columns=('image', 'level'))
    df.to_csv(options.ofilename, columns=('image','level'))

if __name__ == "__main__":
    main()
