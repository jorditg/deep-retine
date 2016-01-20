#!/usr/bin/python

import sys
import numpy
import Image as im


filename = sys.argv[1]

image = im.open(filename)

A = numpy.ndarray((image.size[1], image.size[0], 3), 'u1', image.tostring())

Rmean = numpy.mean(A[:,:,0])
Gmean = numpy.mean(A[:,:,1])
Bmean = numpy.mean(A[:,:,2])

print(filename + "," + str(Rmean) + "," + str(Gmean) + "," + str(Bmean))

