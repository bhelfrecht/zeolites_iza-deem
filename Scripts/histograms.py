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
parser.add_argument('-idxsdd', type=str, default='FPS.idxs', 
        help='File with FPS indices for DEEM-DEEM distances')
parser.add_argument('-idxsd', type=str, default='FPS.idxs', 
        help='File with FPS indices for DEEM distances')
parser.add_argument('-idxsi', type=str, default='FPS.idxs', 
        help='File with FPS indices for IZA distances')
parser.add_argument('-kernel', type=str, default='linear',
        choices=['linear', 'gaussian', 'laplacian'], help='Kernel type')
parser.add_argument('-output', type=str, default='.',
        help='Directory where the output files should be saved')

args = parser.parse_args()

### DISTANCE COMPUTATION ###

# Read SOAP vectors from file
repIdxsDD = np.loadtxt(args.idxsdd, usecols=0, dtype=np.int)
repIdxsD = np.loadtxt(args.idxsd, usecols=0, dtype=np.int)
repIdxsI = np.loadtxt(args.idxsi, usecols=0, dtype=np.int)

# Sample DEEM environments
sys.stdout.write('Reading SOAPs A...\n')
inputFilesD = SOAPTools.read_input(args.deem)
SOAPsD = SOAPTools.build_repSOAPs(inputFilesD, repIdxsD)
SOAPsDD = SOAPTools.build_repSOAPs(inputFilesD, repIdxsDD)

# Read all IZA environments
# (Assume small enough to fit in one batch)
sys.stdout.write('Reading SOAPs B...\n')
inputFilesI = SOAPTools.read_input(args.iza)
SOAPsI = SOAPTools.build_repSOAPs(inputFilesI, repIdxsI)

sys.stdout.write('Computing kernel distance...\n')

# Compute DEEM-DEEM kernel
kii = SOAPTools.build_kernel(SOAPsDD, SOAPsDD, kernel=args.kernel, 
        zeta=args.zeta, width=args.width, nc=None)

# Compute IZA-IZA kernel
kjj = SOAPTools.build_kernel(SOAPsI, SOAPsI, kernel=args.kernel,
        zeta=args.zeta, width=args.width, nc=None)

# Compute DEEM-IZA kernel
kij = SOAPTools.build_kernel(SOAPsD, SOAPsI, kernel=args.kernel,
        zeta=args.zeta, width=args.width, nc=None)

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

# Change kii and kjj for the DEEM-IZA histogram
# Dot products of the diagonal
if args.kernel == 'linear':
    kii = np.sum(SOAPsD*SOAPsD, axis=1)**args.zeta
    kjj = np.sum(SOAPsI*SOAPsI, axis=1)**args.zeta

# Laplacian and Gaussian kernels give 1
# for a kernel between a point and itself
else:
    kii = np.ones(SOAPsD.shape[0])
    kjj = np.ones(SOAPsI.shape[0])
    
print kij.shape

# Histogram of full kernel between DEEM and IZA
D = SOAPTools.kernel_distance(kii, kjj, kij)
H, binEdges = SOAPTools.kernel_histogram_rectangular(D, bins=args.nbins)
np.savetxt('%s/distAB.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

# Min over DEEM: For each IZA point, distance to nearest DEEM point
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, axis=0) 
np.savetxt('%s/minDistBA.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

# Min over IZA: For each DEEM point, distance to nearest IZA point
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, axis=1) 
np.savetxt('%s/minDistAB.hist' % args.output, np.column_stack((binEdges[0:-1], H)))
