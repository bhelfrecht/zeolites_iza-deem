#!/usr/bin/python

import os
import sys
import argparse
import random
import quippy as qp
import numpy as np
import SOAPTools

parser = argparse.ArgumentParser()
parser.add_argument('-soap', type=str, default='SOAPFiles.dat', 
        help='File containing SOAP file filenames')
parser.add_argument('-pca', type=int, default=None, 
        help='Number of PCA components')
parser.add_argument('-kernel', type=str, default='linear',
        choices=['linear', 'gaussian', 'laplacian'], help='Kernel type')
parser.add_argument('-zeta', type=float, default=1, help='SOAP kernel zeta')
parser.add_argument('-width', type=float, default=1.0,
        help='Kernel width')
parser.add_argument('-lowmem', action='store_true',
        help='Low memory version of KPCA')
parser.add_argument('-idxs', type=str, default='FPS.idxs', 
        help='File with FPS indices for representative environments')
parser.add_argument('-dotransform', type=str, default=None,
        help='Project data based on existing kernel')
parser.add_argument('-w', type=str, default=None,
        help='Eigenvectors (as columns) on which to project the new data')
parser.add_argument('-g', type=str, default=None,
        help='G files from KPCA training')
parser.add_argument('-mean', type=str, default=None,
        help='File containing total mean of G')
parser.add_argument('-output', type=str, default='.',
        help='Directory where the output files should be saved')

args = parser.parse_args()

### BUILD PCA ###
# Read inputs
repIdxs = np.loadtxt(args.idxs, dtype=np.int)
inputFiles = SOAPTools.read_input(args.soap)

# Do KPCA projection with existing model
if args.dotransform is not None:
    testFiles = SOAPTools.read_input(args.dotransform)
    eigenvectors = SOAPTools.read_input(args.w)
    gfiles = SOAPTools.read_input(args.g)
    SOAPTools.sparse_kPCA_transform(inputFiles, testFiles, 
            args.mean, gfiles, eigenvectors,
            kernel=args.kernel, zeta=args.zeta, width=args.width,
            nPCA=args.pca, lowmem=args.lowmem, output=args.output)

# Do KPCA construction and projection
else:
    SOAPTools.sparse_kPCA(inputFiles, repIdxs, kernel=args.kernel,
            zeta=args.zeta, width=args.width, nPCA=args.pca,
            lowmem=args.lowmem, output=args.output)
