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
parser.add_argument('-idxsd', type=str, default='FPS.idxs', 
        help='File with FPS indices of unique DEEM environments')
parser.add_argument('-idxsi', type=str, default='FPS.idxs', 
        help='File with FPS indices for unique DEEM environments')
parser.add_argument('-kernel', type=str, default='linear',
        choices=['linear', 'gaussian', 'laplacian'], 
        help='Kernel type')
parser.add_argument('-batchsize', type=int, default=5000,
        help='Batch size for DEEM-DEEM kernel and distance computation')
parser.add_argument('-range', type=float, nargs=2,
        help='Histogram range as `min max`')
parser.add_argument('-output', type=str, default='.',
        help='Directory where the output files should be saved')

args = parser.parse_args()

# Read SOAP vectors from file
repIdxsD = np.loadtxt(args.idxsd, usecols=0, dtype=np.int)
repIdxsI = np.loadtxt(args.idxsi, usecols=0, dtype=np.int)

# Sample DEEM environments
sys.stdout.write('Reading SOAPs A...\n')
inputFilesD = SOAPTools.read_input(args.deem)
SOAPsD = SOAPTools.build_repSOAPs(inputFilesD, repIdxsD)

# Sample IZA environments
sys.stdout.write('Reading SOAPs B...\n')
inputFilesI = SOAPTools.read_input(args.iza)
SOAPsI = SOAPTools.build_repSOAPs(inputFilesI, repIdxsI)

## Compute DEEM-DEEM kernel
sys.stdout.write('Computing DEEM kernel diagonal...\n')

# Batch the environments
n_batches = SOAPsD.shape[0]/args.batchsize
if SOAPsD.shape[0] % args.batchsize != 0:
    n_batches += 1

# Keep track of the diagonal index, as it will change as
# we work through the batches
b = 0

# Set up the diagonal, i.e., K(A, A)
kii_diag = np.zeros(SOAPsD.shape[0])

# Compute the kernel for each batch will all of the unique environments
for n in range(0, n_batches):
    sys.stdout.write('Batch %d\n' % (n+1))

    # Compute kernel
    kiib = SOAPTools.build_kernel(SOAPsD[b:b+args.batchsize], SOAPsD, 
            kernel=args.kernel, zeta=args.zeta, width=args.width, nc=None)

    # Extract diagonal of kernel, i.e., K(A, A)
    kii_diag[b:b+args.batchsize] = np.diag(kiib, b)

    # Increment the diagonal
    b += args.batchsize

## Compute DEEM-DEEM histograms
sys.stdout.write('Computing DEEM kernel distances...\n')

# Reset the diagonal index
b = 0

# Initialize the histogram values
H = np.zeros(args.nbins)
Hmin = np.zeros(args.nbins)

# Re-compute the kernel for each batch rather than saving
# a bunch of ~ 3GB binary files. This is relatively speedy
# for linear kernels, but Gaussian kernels might take a while
for n in range(0, n_batches):
    sys.stdout.write('Batch %d\n' % (n+1))

    # Compute kernel
    kiib = SOAPTools.build_kernel(SOAPsD[b:b+args.batchsize], SOAPsD, 
            kernel=args.kernel, zeta=args.zeta, width=args.width, nc=None)

    # Compute squared distance
    D = -2.0*kiib + np.reshape(kii_diag[b:b+args.batchsize], (-1, 1)) + kii_diag

    # Offset for taking the upper triangle of the distance matrix 
    offset = b

    # Deal with machine precision errors around zero
    # (sometimes we get very tiny negative distances)
    D[D < 0.0] = 0.0

    # Compute distance
    D = np.sqrt(D)

    # Take upper triangular of distance matrix, so we can determine
    # the min distance for each environment in the batch
    Dmin = np.triu(D, k=offset)

    # Throw out zero distances in the distance matrix
    D = D[D > 0.0]

    # Set zero distances in the min distance matrix
    # to a large value so we don't just end up with a bunch
    # of zero minimum distances
    Dmin[Dmin == 0] = args.range[1]**2

    # Compute minimum distance for each environment in the batch
    Dmin = np.amin(Dmin, axis=1)

    # Compute non-normalized histogram on the distances
    h, binEdges = np.histogram(D, bins=args.nbins, 
            range=args.range, density=False)

    # Compute non-normalized histogram on the minimum distances
    hmin, binEdges = np.histogram(Dmin, bins=args.nbins, 
            range=args.range, density=False)

    # Accumulate the samples in each bin
    H += h
    Hmin += hmin

    # Increment the diagonal index
    b += args.batchsize

# Normalize the histograms
dx = (args.range[1]-args.range[0])/args.nbins
H /= np.sum(H)*dx
Hmin /= np.sum(Hmin)*dx

# Save the histograms
np.savetxt('%s/distAA.hist' % args.output, np.column_stack((binEdges[0:-1], H)))
np.savetxt('%s/minDistAA.hist' % args.output, np.column_stack((binEdges[0:-1], Hmin)))
    
## Compute IZA-IZA kernel
sys.stdout.write('Buildilng IZA-IZA kernel...\n')
kjj = SOAPTools.build_kernel(SOAPsI, SOAPsI, kernel=args.kernel,
        zeta=args.zeta, width=args.width, nc=None)

## Compute DEEM-IZA kernel
sys.stdout.write('Buildilng DEEM-IZA kernel...\n')
kij = SOAPTools.build_kernel(SOAPsD, SOAPsI, kernel=args.kernel,
        zeta=args.zeta, width=args.width, nc=None)

## Histogram of full kernel between IZA and IZA
sys.stdout.write('Computing IZA-IZA histogram...\n')
# Compute IZA-IZA distance matrix
D = SOAPTools.kernel_distance(np.diag(kjj), np.diag(kjj), kjj)

# Compute normalized IZA-IZA histogram
H, binEdges = SOAPTools.kernel_histogram_square(D, bins=args.nbins, range=args.range)
np.savetxt('%s/distBB.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

## Min over IZA: Min distance from each IZA point to a IZA point
sys.stdout.write('Computing IZA-IZA min histogram...\n')

# Set the K(B, B) distances to a large value so we don't
# get a bunch of zero minimum distances
np.fill_diagonal(D, args.range[1]**2)

# Compute normalized IZA-IZA histogram
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, range=args.range, axis=0) 
np.savetxt('%s/minDistBB.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

## Histogram of full kernel between DEEM and IZA
sys.stdout.write('Computing DEEM-IZA histogram...\n')

# Compute DEEM-IZA distance matrix
D = SOAPTools.kernel_distance(kii_diag, np.diag(kjj), kij)

# Compute normalized DEEM-IZA histogram
H, binEdges = SOAPTools.kernel_histogram_rectangular(D, bins=args.nbins, range=args.range)
np.savetxt('%s/distAB.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

## Min over DEEM: For each IZA point, distance to nearest DEEM point
sys.stdout.write('Computing IZA-DEEM min histogram...\n')

# Compute normalized IZA-DEEM histogram
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, range=args.range, axis=0) 
np.savetxt('%s/minDistBA.hist' % args.output, np.column_stack((binEdges[0:-1], H)))

## Min over IZA: For each DEEM point, distance to nearest IZA point
sys.stdout.write('Computing DEEM-IZA min histogram...\n')

# Compute the normalized DEEM-IZA histogram
H, binEdges = SOAPTools.kernel_histogram_min(D, bins=args.nbins, range=args.range, axis=1) 
np.savetxt('%s/minDistAB.hist' % args.output, np.column_stack((binEdges[0:-1], H)))
