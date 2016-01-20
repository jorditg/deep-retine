#!/usr/bin/python

import sys
import numpy
import Image as im


filename = sys.argv[1]

Ymean = float(sys.argv[2])
Umean = float(sys.argv[3])
Vmean = float(sys.argv[4])

image = im.open(filename)
ycbcr = image.convert('YCbCr')

B = numpy.ndarray((image.size[1], image.size[0], 3), 'u1', ycbcr.tostring())

Yvar = numpy.mean(numpy.square(B[:,:,0] - Ymean))
Uvar = numpy.mean(numpy.square(B[:,:,1] - Umean))
Vvar = numpy.mean(numpy.square(B[:,:,2] - Vmean))

print(filename + "," + str(Yvar) + "," + str(Uvar) + "," + str(Vvar))

