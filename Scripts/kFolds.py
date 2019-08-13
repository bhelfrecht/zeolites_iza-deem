#!/usr/bin/python

import os
import sys
import argparse
import random
import quippy as qp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=float, default=1.0, 
        help='Fraction of data to use for k-fold construction')
parser.add_argument('-nt', type=int, default=None, 
        help='Total number of data points')
parser.add_argument('-k', type=int, default=5, help='Number of folds')
parser.add_argument('-output', type=str, default='.',
        help='Directory in which to save output files')

args = parser.parse_args()

# Shuffle all indices
structureIdxs = np.arange(0, args.nt)
np.random.shuffle(structureIdxs)
randomIdxs = structureIdxs[0:int(args.nt*args.f)]
testIdxs = structureIdxs[int(args.nt*args.f):]
np.savetxt('%s/test.idxs' % args.output, np.sort(testIdxs), fmt='%d')

foldSize = int(float(len(randomIdxs))/args.k)

# Fill the validation and train sets
f = open('%s/kValidate.idxs' % args.output, 'w')
g = open('%s/kTrain.idxs' % args.output, 'w')
n = 0
for i in range(0, args.k):
    validationIdxs = np.sort(randomIdxs[n:n+foldSize])
    trainIdxs = np.setdiff1d(randomIdxs, validationIdxs) # already sorted
    np.savetxt(f, np.reshape(validationIdxs, (1, len(validationIdxs))), fmt='%d')
    np.savetxt(g, np.reshape(trainIdxs, (1, len(trainIdxs))), fmt='%d')
    n += foldSize
