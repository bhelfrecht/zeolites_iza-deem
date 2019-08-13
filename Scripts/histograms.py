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
        help='SOAP kernel zeta')
parser.add_argument('-nbins', type=int, default=200, 
        help='Number of histogram bins')
parser.add_argument('-idxs', type=str, default='FPS.idxs', 
        help='File with FPS indices')

args = parser.parse_args()

### DISTANCE COMPUTATION ###

# Read SOAP vectors from file
idxsRef = np.loadtxt(args.idxs, dtype=np.int, unpack=True)
if idxsRef.ndim > 1:
    batchIDs = idxsRef[1]
    idxsRef = idxsRef[0]

# Sample DEEM environments
sys.stdout.write('Reading SOAPs A...\n')
f = open(args.deem, 'r')
inputFiles = f.readlines()
inputFiles = [i.strip() for i in inputFiles]
f.close()
SOAPsA = []
for idx, i in enumerate(inputFiles):
    idxsRep = idxsRef[np.where(batchIDs == idx)]
    SOAPsA.append(np.load(i)[idxsRep])
SOAPsA = np.concatenate(SOAPsA)

# Read all IZA environments
sys.stdout.write('Reading SOAPs B...\n')
g = open(args.iza, 'r')
inputFiles = g.readlines()
inputFiles = [i.strip() for i in inputFiles]
g.close()
SOAPsB = []
for i in inputFiles:
    SOAPsB.append(np.load(i))
SOAPsB = np.concatenate(SOAPsB)

sys.stdout.write('Computing kernel distance...\n')

kii = SOAPTools.build_kernel(SOAPsA, SOAPsA, zeta=args.zeta)
kjj = SOAPTools.build_kernel(SOAPsB, SOAPsB, zeta=args.zeta)
kij = SOAPTools.build_kernel(SOAPsA, SOAPsB, zeta=args.zeta)

sys.stdout.write('Computing histograms...\n')

# Histogram of full kernel between DEEM and DEEM
D = SOAPTools.kernel_distance(np.diag(kii), np.diag(kii), kii)
H, binEdges = SOAPTools.kernel_histogram_square(D, bins=args.nbins)
np.savetxt('distAA.hist', np.column_stack((binEdges[0:-1], H)))

# Min over DEEM: Min distance from each DEEM point to a DEEM point
np.fill_diagonal(D, 1.0)
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, axis=0) 
np.savetxt('minDistAA.hist', np.column_stack((binEdges[0:-1], H)))

# Histogram of full kernel between IZA and IZA
D = SOAPTools.kernel_distance(np.diag(kjj), np.diag(kjj), kjj)
H, binEdges = SOAPTools.kernel_histogram_square(D, bins=args.nbins)
np.savetxt('distBB.hist', np.column_stack((binEdges[0:-1], H)))

# Min over IZA: Min distance from each IZA point to a IZA point
np.fill_diagonal(D, 1.0)
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, axis=0) 
np.savetxt('minDistBB.hist', np.column_stack((binEdges[0:-1], H)))

# Histogram of full kernel between DEEM and IZA
D = SOAPTools.kernel_distance(np.diag(kii), np.diag(kjj), kij)
H, binEdges = SOAPTools.kernel_histogram_rectangular(D, bins=args.nbins)
np.savetxt('distAB.hist', np.column_stack((binEdges[0:-1], H)))

# Min over DEEM: For each IZA point, distance to nearest DEEM point
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, axis=0) 
np.savetxt('minDistBA.hist', np.column_stack((binEdges[0:-1], H)))

# Min over IZA: For each DEEM point, distance to nearest IZA point
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, axis=1) 
np.savetxt('minDistAB.hist', np.column_stack((binEdges[0:-1], H)))
