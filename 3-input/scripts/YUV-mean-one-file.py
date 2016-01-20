#!/usr/bin/python

import sys
import numpy
import Image as im


filename = sys.argv[1]

image = im.open(filename)
ycbcr = image.convert('YCbCr')

B = numpy.ndarray((image.size[1], image.size[0], 3), 'u1', ycbcr.tostring())

Ymean = numpy.mean(B[:,:,0])
Umean = numpy.mean(B[:,:,1])
Vmean = numpy.mean(B[:,:,2])

print(filename + "," + str(Ymean) + "," + str(Umean) + "," + str(Vmean))

