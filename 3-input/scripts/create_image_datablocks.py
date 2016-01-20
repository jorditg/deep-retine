#!/usr/bin/python

import sys, getopt

def usage():
    print "This script prepares a set of data blocks to train a neural network"
    

def main(argv):
    float_size = 4
    image_channels = 3
    image_height = 512
    image_width = 512
    GPU_datablock_size_limit_GB = 3
    image_enable_random_rotation = False
    image_enable_random_horizontal_mirroring = False
    image_enable_random_vertical_mirroring = False

    try:
        opts, args = getopt.getopt(argv, "h", ["float_size=", "channels=", "height=", "width=", "block_limit_size_GB=", "rotation", "horizontal_mirroring", "vertical_mirroring"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == '--float_size':
            float_size = int(arg)
        elif opt == "--channels":
            image_channels = int(arg)
        elif opt == "--height":
            image_height = int(arg)
        elif opt == "--width":
            image_width = int(arg)
        elif opt == "--block_limit_size_GB":
            GPU_datablock_size_limit_GB = int(arg)
        elif opt == "--rotation":
            image_enable_random_rotation = True
        elif opt == "--horizontal_mirroring":
            image_enable_random_horizontal_mirroring = True
        elif opt == "--vertical_mirroring":
            image_enable_random_vertical_mirroring = True
        
