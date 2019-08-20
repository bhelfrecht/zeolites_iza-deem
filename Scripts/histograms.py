#!/usr/bin/python

import os
import sys
import argparse
import numpy as np
import SOAPTools

parser = argparse.ArgumentParser()
parser.add_argument('-deem', type=str, default='SOAPFiles.dat',
        help='File containing SOAP file filenames for DEEM')
parser.add_argument('-iza', type=str, default='SOAPFiles.dat',
        help='File containing SOAP file filenames for IZA')
parser.add_argument('-zeta', type=float, default=1, 
        help='linear kernel zeta')
parser.add_argument('-width', type=float, default=0.1,
        help='Gaussian kernel width')
parser.add_argument('-nbins', type=int, default=200, 
        help='Number of histogram bins')
parser.add_argument('-idxs', type=str, default='FPS.idxs', 
        help='File with FPS indices')
parser.add_argument('-kernel', type=str, default='linear',
        choices=['linear', 'gaussian', 'laplacian'], help='Kernel type')
parser.add_argument('-npca', type=int, default=None,
        help='Number of (PCA) components to use')
parser.add_argument('-output', type=str, default='.',
        help='Directory where the output files should be saved')

args = parser.parse_args()

### DISTANCE COMPUTATION ###

# Read SOAP vectors from file
repIdxs = np.loadtxt(args.idxs, dtype=np.int)

# Sample DEEM environments
sys.stdout.write('Reading SOAPs A...\n')
inputFiles = SOAPTools.read_input(args.deem)
SOAPsA = SOAPTools.build_repSOAPs(inputFiles, repIdxs)

# Read all IZA environments
sys.stdout.write('Reading SOAPs B...\n')
inputFiles = SOAPTools.read_input(args.iza)
SOAPsB = []
for i in inputFiles:
    SOAPsB.append(np.load(i))
SOAPsB = np.concatenate(SOAPsB)

sys.stdout.write('Computing kernel distance...\n')

kii = SOAPTools.build_kernel(SOAPsA, SOAPsA, kernel=args.kernel, 
        zeta=args.zeta, width=args.width, nc=args.npca)
kjj = SOAPTools.build_kernel(SOAPsB, SOAPsB, kernel=args.kernel,
        zeta=args.zeta, width=args.width, nc=args.npca)
kij = SOAPTools.build_kernel(SOAPsA, SOAPsB, kernel=args.kernel,
        zeta=args.zeta, width=args.width, nc=args.npca)

sys.stdout.write('Computing histograms...\n')

# Histogram of full kernel between DEEM and DEEM
D = SOAPTools.kernel_distance(np.diag(kii), np.diag(kii), kii)
H, binEdges = SOAPTools.kernel_histogram_square(D, bins=args.nbins)
np.savetxt('%s/distAA.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

# Min over DEEM: Min distance from each DEEM point to a DEEM point
np.fill_diagonal(D, 1.0)
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, axis=0) 
np.savetxt('%s/minDistAA.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

# Histogram of full kernel between IZA and IZA
D = SOAPTools.kernel_distance(np.diag(kjj), np.diag(kjj), kjj)
H, binEdges = SOAPTools.kernel_histogram_square(D, bins=args.nbins)
np.savetxt('%s/distBB.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

# Min over IZA: Min distance from each IZA point to a IZA point
np.fill_diagonal(D, 1.0)
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, axis=0) 
np.savetxt('%s/minDistBB.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

# Histogram of full kernel between DEEM and IZA
D = SOAPTools.kernel_distance(np.diag(kii), np.diag(kjj), kij)
H, binEdges = SOAPTools.kernel_histogram_rectangular(D, bins=args.nbins)
np.savetxt('%s/distAB.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

# Min over DEEM: For each IZA point, distance to nearest DEEM point
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, axis=0) 
np.savetxt('%s/minDistBA.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

# Min over IZA: For each DEEM point, distance to nearest IZA point
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, axis=1) 
np.savetxt('%s/minDistAB.hist' % args.output, np.column_stack((binEdges[0:-1], H)))
