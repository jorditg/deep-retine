#!/usr/bin/python

import sys
import numpy
import Image as im


filename = sys.argv[1]

Rmean = float(sys.argv[2])
Gmean = float(sys.argv[3])
Bmean = float(sys.argv[4])

image = im.open(filename)

A = numpy.ndarray((image.size[1], image.size[0], 3), 'u1', image.tostring())

Rvar = numpy.mean(numpy.square(A[:,:,0] - Rmean))
Gvar = numpy.mean(numpy.square(A[:,:,1] - Gmean))
Bvar = numpy.mean(numpy.square(A[:,:,2] - Bmean))

print(filename + "," + str(Rvar) + "," + str(Gvar) + "," + str(Bvar))

