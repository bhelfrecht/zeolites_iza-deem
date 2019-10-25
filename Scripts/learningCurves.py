#!/usr/bin/python

import os
import sys
import argparse
import quippy as qp
import numpy as np
from scipy.spatial.distance import cdist
import SOAPTools

parser = argparse.ArgumentParser()
parser.add_argument('-structure', type=str, default=None, 
        help='File containing structures')
parser.add_argument('-soap', type=str, default='SOAPFiles.dat',
        help='File containing SOAP file filenames')
parser.add_argument('-idxs', type=str, default='FPS.idxs', 
        help='File with FPS indices')
parser.add_argument('-p', type=str, default='volume', 
        choices=['volume', 'Energy_per_Si', 'Energy_per_Si_Opt'], 
        help='Property name for regression')
parser.add_argument('-Z', type=int, nargs='+', default=None, 
        help='Space separated atomic numbers of center species')
parser.add_argument('-k', type=int, default=5, 
        help='Number of folds for cross validation')
parser.add_argument('-kernel', type=str, default='linear', 
        choices=['linear', 'gaussian', 'laplacian'], help='Kernel type')
parser.add_argument('-zeta', type=float, default=1, help='SOAP kernel zeta')
parser.add_argument('-sigma', type=float, default=[1.0], nargs='+',
        help='Regularization sigma for regression')
parser.add_argument('-width', type=float, default=[None], nargs='+', 
        help='Kernel width')
parser.add_argument('-j', type=float, default=[1.0E-16], nargs='+',
        help='Value of "sparse jitter"')
parser.add_argument('-pcalearn', type=int, default=[None], nargs='+',
        help='List of numbers of principal components used to build the kernel')
parser.add_argument('-ntrain', type=int, default=[0], nargs='+',
        help='Number of training structures for property regression')
parser.add_argument('-train', type=str, default=None,
        help='File containing k-fold training indices')
parser.add_argument('-test', type=str, default=None,
        help='File containing k-fold testing indices')
parser.add_argument('-validate', type=str, default=None,
        help='File containing k-fold validation indices')
parser.add_argument('-env', action='store_true', 
        help='Compute property decomposition into environment contributions')
parser.add_argument('-lowmem', action='store_true',
        help='Compute kernel for environment decomposition in batches')
parser.add_argument('-shuffle', action='store_true',
        help='Do an extra shuffle in the learning curves, so '\
                'training sets at increasing numbers of training points '\
                'are independent')
parser.add_argument('-output', type=str, default='.',
        help='Directory where the output files should be placed')

args = parser.parse_args()

# Remove iterations over kernel width for linear kernels
if args.kernel == 'linear':
    args.width = [None]

### PROPERTY EXTRACTION ###
# Extract structure properties
sys.stdout.write('Extracting properties...\n')
al = qp.AtomsReader(args.structure)
structIdxs, nAtoms, volume, p \
        = SOAPTools.extract_structure_properties(al, args.Z, propName=args.p)

np.savetxt('%s/props.dat' % args.output, p) #####

# Scale property
if args.p == 'volume':
     p /= nAtoms/3 

# Energies
else:

    # Convert to total energy
    p *= nAtoms/3

    # Remove mean binding energy
    p -= np.mean(p/nAtoms)*nAtoms

    # Convert back to energy per Si
    p /= nAtoms/3

# Get k-fold indices
trainIdxs = np.loadtxt(args.train, dtype=np.int)
validateIdxs = np.loadtxt(args.validate, dtype=np.int)
if args.test is not None:
    testIdxs = np.loadtxt(args.test, dtype=np.int)

# Shuffle training indices for each iteration
for i in range(0, args.k):
    np.random.shuffle(trainIdxs[i])

# Setup output files
nPCA = len(args.pcalearn)
nWidth = len(args.width)
nSigma = len(args.sigma)
nTrain = len(args.ntrain)
nJitter = len(args.j)
maeTrain = np.zeros((args.k, nPCA, nWidth, nSigma, nJitter, nTrain, 7))
maeTest = np.zeros((args.k, nPCA, nWidth, nSigma, nJitter, nTrain, 7))
rmseTrain = np.zeros((args.k, nPCA, nWidth, nSigma, nJitter, nTrain, 7))
rmseTest = np.zeros((args.k, nPCA, nWidth, nSigma, nJitter, nTrain, 7))
maeAvgTrain = np.zeros((nPCA, nWidth, nSigma, nJitter, nTrain, 7))
maeAvgTest = np.zeros((nPCA, nWidth, nSigma, nJitter, nTrain, 7))
rmseAvgTrain = np.zeros((nPCA, nWidth, nSigma, nJitter, nTrain, 7))
rmseAvgTest = np.zeros((nPCA, nWidth, nSigma, nJitter, nTrain, 7))

# Select representative environments
repIdxs = np.loadtxt(args.idxs, dtype=np.int)

# Read SOAP vectors and extract representative environments
inputFiles = SOAPTools.read_input(args.soap)
repSOAPs = SOAPTools.build_repSOAPs(inputFiles, repIdxs)
SOAPs = []
envKernel = None 

### PROPERTY REGRESSION ###
# Perform property decomposition
sys.stdout.write('Performing property regression...\n')
for idx, i in enumerate(args.pcalearn):
    if i is not None:
        sys.stdout.write('-----> PCA: %d\n' % i)
    for wdx, w in enumerate(args.width):
        if w is not None:
            sys.stdout.write('----> Width: %.2e\n' % w)

        # Build the appropriate kernel
        if args.kernel == 'gaussian':
            dMM = cdist(repSOAPs[:, 0:i], repSOAPs[:, 0:i], metric='euclidean')
            kMM = SOAPTools.gaussianKernel(dMM, w)
            kNM = SOAPTools.build_sum_kernel_batch(inputFiles, repSOAPs[:, 0:i],
                    structIdxs, kernel='gaussian', width=w, nc=i)

            # Build the environment kernel, if desired
            if args.env is True:
                envKernel = SOAPTools.build_kernel_batch(inputFiles, 
                        repSOAPs[:, 0:i], kernel='gaussian', 
                        width=w, nc=i, lowmem=args.lowmem)
        elif args.kernel == 'laplacian':
            dMM = cdist(repSOAPs[:, 0:i], repSOAPs[:, 0:i], metric='cityblock')
            kMM = SOAPTools.laplacianKernel(dMM, w)
            kNM = SOAPTools.build_sum_kernel_batch(inputFiles, repSOAPs[:, 0:i],
                    structIdxs, kernel='laplacian', width=w, nc=i)

            # Build the environment kernel, if desired
            if args.env is True:
                envKernel = SOAPTools.build_kernel_batch(inputFiles, 
                        repSOAPs[:, 0:i], kernel='laplacian', 
                        width=w, nc=i, lowmem=args.lowmem)
        else: # linear kernel
            kMM = SOAPTools.build_kernel(repSOAPs[:, 0:i], repSOAPs[:, 0:i], 
                    zeta=args.zeta)
            kNM = SOAPTools.build_sum_kernel_batch(inputFiles, repSOAPs[:, 0:i],
                    structIdxs, zeta=args.zeta, nc=i)

            # Build the environment kernel, if desired
            if args.env is True:
                envKernel = SOAPTools.build_kernel_batch(inputFiles, 
                        repSOAPs[:, 0:i], zeta=args.zeta, 
                        nc=i, lowmem=args.lowmem)

        # Scale the kernel matrices so that
        # they represent the average kernel
        # over the structure properties
        kNM = (kNM.T*3/nAtoms).T

        # Compute learning curves for each set of hyperparameters
        for sdx, s in enumerate(args.sigma):
            sys.stdout.write('---> Sigma: %.2e\n' % s)
            for jdx, j in enumerate(args.j):
                sys.stdout.write('--> Jitter: %.2e\n' % j)
                for ndx, n in enumerate(args.ntrain):
                    sys.stdout.write('-> No. Training Points: %d\n' % n)
                    for k in range(0, args.k):
                        sys.stdout.write('> Iteration: %d\r' % (k+1))
                        sys.stdout.flush()

                        # Set up the test and training sets
                        if args.shuffle:
                            np.random.shuffle(trainIdxs[k])
                        idxsTrain = trainIdxs[k, 0:n]
                        idxsValidate = validateIdxs[k]

                        # Perform the KRR
                        yTrain, yTest, yyTrain, yyTest, _ \
                                = SOAPTools.property_regression(p, kMM, kNM, 
                                        len(structIdxs), idxsTrain, 
                                        idxsValidate, sigma=s, 
                                        envKernel=envKernel, jitter=j,
                                        output=args.output)

                        # Write out some debug error info
                        #np.savetxt('yTrain-PCA%d-%d-%d.dat' % (i, n, k),
                        #        np.column_stack((yTrain, yyTrain)))
                        #np.savetxt('yTest-PCA%d-%d-%d.dat' % (i, n, k),
                        #        np.column_stack((yTest, yyTest)))

                        # Build the error matrices (see analysis notebooks
                        # for full explanation of the matrix structure)
                        maeTrain[k, idx, wdx, sdx, jdx, ndx, -1] \
                                = SOAPTools.MAE(yyTrain, yTrain)
                        maeTest[k, idx, wdx, sdx, jdx, ndx, -1] \
                                = SOAPTools.MAE(yyTest, yTest)
                        rmseTrain[k, idx, wdx, sdx, jdx, ndx, -1] \
                                = SOAPTools.RMSE(yyTrain, yTrain)
                        rmseTest[k, idx, wdx, sdx, jdx, ndx, -1] \
                                = SOAPTools.RMSE(yyTest, yTest)
    
                        for x in [maeTrain, maeTest, rmseTrain, rmseTest]:
                            for ydx, y in enumerate([k, i, w, s, j, n]):
                                x[k, idx, wdx, sdx, jdx, ndx, ydx] = y
    
                    # Build the average error matrices (see analysis notebooks
                    # for full explanation of the matrix structure)
                    maeAvgTrain[idx, wdx, sdx, jdx, ndx, -2] \
                            = np.mean(maeTrain[:, idx, wdx, sdx, jdx, ndx, -1])
                    maeAvgTest[idx, wdx, sdx, jdx, ndx, -2] \
                            = np.mean(maeTest[:, idx, wdx, sdx, jdx, ndx, -1])
                    rmseAvgTrain[idx, wdx, sdx, jdx, ndx, -2] \
                            = np.mean(rmseTrain[:, idx, wdx, sdx, jdx, ndx, -1])
                    rmseAvgTest[idx, wdx, sdx, jdx, ndx, -2] \
                            = np.mean(rmseTest[:, idx, wdx, sdx, jdx, ndx, -1])
    
                    maeAvgTrain[idx, wdx, sdx, jdx, ndx, -1] \
                            = np.std(maeTrain[:, idx, wdx, sdx, jdx, ndx, -1])
                    maeAvgTest[idx, wdx, sdx, jdx, ndx, -1] \
                            = np.std(maeTest[:, idx, wdx, sdx, jdx, ndx, -1])
                    rmseAvgTrain[idx, wdx, sdx, jdx, ndx, -1] \
                            = np.std(rmseTrain[:, idx, wdx, sdx, jdx, ndx, -1])
                    rmseAvgTest[idx, wdx, sdx, jdx, ndx, -1] \
                            = np.std(rmseTest[:, idx, wdx, sdx, jdx, ndx, -1])
    
                    for x in [maeAvgTrain, maeAvgTest, 
                            rmseAvgTrain, rmseAvgTest]:
                        for ydx, y in enumerate([i, w, s, j, n]):
                            x[idx, wdx, sdx, jdx, ndx, ydx] = y
                    sys.stdout.write('\n')

# Save error matrices to file
np.save('%s/maeAvgTrain.npy' % args.output, maeAvgTrain)
np.save('%s/maeAvgTest.npy' % args.output, maeAvgTest)
np.save('%s/rmseAvgTrain.npy' % args.output, rmseAvgTrain)
np.save('%s/rmseAvgTest.npy' % args.output, rmseAvgTest)

np.save('%s/maeTrain.npy' % args.output, maeTrain)
np.save('%s/maeTest.npy' % args.output, maeTest)
np.save('%s/rmseTrain.npy' % args.output, rmseTrain)
np.save('%s/rmseTest.npy' % args.output, rmseTest)
