#!/usr/bin/python

import os
import sys
import argparse
#import quippy as qp
import ase.io as aseIO
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
parser.add_argument('-kernel', type=str, default='linear', 
        choices=['linear', 'gaussian', 'laplacian'], 
        help='Kernel type')
parser.add_argument('-zeta', type=float, default=1, 
        help='SOAP kernel zeta')
parser.add_argument('-sigma', type=float, default=1.0,
        help='Regularization sigma for regression')
parser.add_argument('-width', type=float, default=1.0, 
        help='Kernel width')
parser.add_argument('-j', type=float, default=1.0E-16,
        help='Value of sparse jitter')
parser.add_argument('-npca', type=int, default=None,
        help='Number of principal components used to build the kernel')
parser.add_argument('-ntrain', type=int, default=0,
        help='Number of training structures for property regression')
parser.add_argument('-env', action='store_true', 
        help='Compute property decomposition into environment contributions')
parser.add_argument('-lowmem', action='store_true',
        help='Compute kernel for environment decomposition in batches')
parser.add_argument('-output', type=str, default='.',
        help='Directory where the output files should be saved')
parser.add_argument('-project', type=str, default=None,
        help='File containing SOAP file filenames for projection')
parser.add_argument('-w', type=str, default=None,
        help='File with regression weights for projection')
parser.add_argument('-mean', type=str, default=None,
        help='File with property mean for projection')
parser.add_argument('-kref', type=str, default=None,
        help='File containing the reference kernel for centering')
parser.add_argument('-test', type=str, default=None,
        help='File containing indices for testing')

args = parser.parse_args()

if args.w is None or not args.env:

    # Extract structure properties
    sys.stdout.write('Extracting properties...\n')
    #al = qp.AtomsReader(args.structure)
    al = aseIO.read(args.structure, index=':')
    structIdxs, nAtoms, volume, p \
            = SOAPTools.extract_structure_properties(al, args.Z, propName=args.p)
        
# Do regression if we are not 
# given regression weights 
if args.w is None:

    # Train-test split if no test indices supplied
    if args.test is None:
        randomIdxs = np.arange(0, len(structIdxs))
        np.random.shuffle(randomIdxs)
        trainIdxs = randomIdxs[0:args.ntrain]
        testIdxs = randomIdxs[args.ntrain:]

    # Train-test split with test indices supplied
    else:
        testIdxs = np.loadtxt(args.test, dtype=np.int)
        trainIdxs = np.arange(0, len(structIdxs))
        trainIdxs = np.delete(trainIdxs, testIdxs)

    # Scale property
    if args.p == 'volume':
        p /= nAtoms/3
        p_mean = np.mean(p[trainIdxs])
        #p[trainIdxs] -= p_mean
        #p[testIdxs] -= p_mean
        p -= p_mean
    
    # Energies
    else:
    
        # Convert to total energy
        p *= nAtoms/3
        p_mean = np.mean(p[trainIdxs]/nAtoms[trainIdxs])

        # Remove mean binding energy
        #p[trainIdxs] -= p_mean*nAtoms[trainIdxs]
        #p[testIdxs] -= p_mean*nAtoms[testIdxs]
        p -= p_mean*nAtoms

        # Convert back to energy per Si
        p /= nAtoms/3

    # Save mean property
    g = open('%s/property_mean.dat' % args.output, 'w')
    g.write('%.16f\n' % p_mean)
    g.close()

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

    # Build desired kernels
    if args.kernel == 'gaussian':
        kMM = SOAPTools.build_kernel(repSOAPs[:, 0:args.npca],
                repSOAPs[:, 0:args.npca], kernel='gaussian', width=args.width)
        kNM = SOAPTools.build_sum_kernel_batch(inputFiles, 
                repSOAPs[:, 0:args.npca],
                structIdxs, kernel='gaussian', width=args.width, nc=args.npca)

        # Build environment kernel, if required
        if args.env is True:
            envKernel = SOAPTools.build_kernel_batch(inputFiles, 
                    repSOAPs[:, 0:args.npca], 
                    kernel='gaussian', width=args.width, 
                    nc=args.npca, lowmem=args.lowmem, output=args.output)

    elif args.kernel == 'laplacian':
        kMM = SOAPTools.build_kernel(repSOAPs[:, 0:args.npca],
                repSOAPs[:, 0:args.npca], kernel='gaussian', width=args.width)
        kNM = SOAPTools.build_sum_kernel_batch(inputFiles, 
                repSOAPs[:, 0:args.npca],
                structIdxs, kernel='laplacian', width=args.width, nc=args.npca)

        # Build environment kernel, if required
        if args.env is True:
            envKernel = SOAPTools.build_kernel_batch(inputFiles, 
                    repSOAPs[:, 0:args.npca],
                    kernel='laplacian', width=args.width, 
                    nc=args.npca, lowmem=args.lowmem, output=args.output)

    else: # linear kernel
        kMM = SOAPTools.build_kernel(repSOAPs[:, 0:args.npca], 
                repSOAPs[:, 0:args.npca], kernel='linear', zeta=args.zeta)
        kNM = SOAPTools.build_sum_kernel_batch(inputFiles, 
                repSOAPs[:, 0:args.npca], 
                structIdxs, zeta=args.zeta, nc=args.npca)

        # Build environment kernel, if required
        if args.env is True:
            envKernel = SOAPTools.build_kernel_batch(inputFiles, 
                    repSOAPs[:, 0:args.npca],
                    zeta=args.zeta, nc=args.npca, 
                    lowmem=args.lowmem, output=args.output)
    
    kNM = (kNM.T*3/nAtoms).T

    # Center kernel
    #kMM = SOAPTools.center_kernel(kMM)
    #kNM = SOAPTools.center_kernel(kNM, Kref=kMM)
    #np.save('%s/kMM' % args.output, kMM)

    # Header for the output file with parameter information
    # about the regression
    header = 'Kernel = %s, Width = %1.3E, Zeta = %1.3E, '\
            'Sigma = %1.3E, nPCA = %s, nTrain = %d, '\
            'Property = %s' % (args.kernel, args.width, args.zeta,
                    args.sigma, args.npca, args.ntrain, args.p)

    # Perform KRR
    yTrain, yTest, yyTrain, yyTest, w \
            = SOAPTools.property_regression(p, kMM, kNM, 
                    len(structIdxs), trainIdxs, testIdxs, 
                    sigma=args.sigma, jitter=args.j, 
                    envKernel=envKernel, output=args.output)
    
    # Save regression output as [True value, predicted value]
    np.savetxt('%s/yTrain.dat' % args.output, 
            np.column_stack((yTrain, yyTrain)), header=header)
    np.savetxt('%s/yTest.dat' % args.output, 
            np.column_stack((yTest, yyTest)), header=header)

    # Save weights
    np.savetxt('%s/w.dat' % args.output, w)

# Projecting from provided weights
else:
    sys.stdout.write('Projecting properties...\n')

    # Load weights
    w = np.loadtxt(args.w)

    # Load SOAPs
    inputFiles = SOAPTools.read_input(args.soap)
    repIdxs = np.loadtxt(args.idxs, dtype=np.int)
    repSOAPs = SOAPTools.build_repSOAPs(inputFiles, repIdxs)
    projFiles = SOAPTools.read_input(args.project)

    # Reference kernel for centering
    #k_ref = np.load(args.kref)

    # Build environment kernels
    if args.env:
        if args.kernel == 'gaussian':
            k = SOAPTools.build_kernel_batch(projFiles, 
                    repSOAPs[:, 0:args.npca], 
                    kernel='gaussian', width=args.width, 
                    nc=args.npca, lowmem=args.lowmem, output=args.output)
        elif args.kernel == 'laplacian':
            k = SOAPTools.build_kernel_batch(projFiles, 
                    repSOAPs[:, 0:args.npca],
                    kernel='laplacian', width=args.width, 
                    nc=args.npca, lowmem=args.lowmem, output=args.output)
        else: # linear kernel
            k = SOAPTools.build_kernel_batch(projFiles, 
                    repSOAPs[:, 0:args.npca],
                    zeta=args.zeta, nc=args.npca, 
                    lowmem=args.lowmem, output=args.output)

        # Center environment kernels
        #if isinstance(k, list):
        #    for idx, i in enumerate(k):
        #        ik = read_SOAP('%s.npy' % i)
        #        ik = SOAPTools.center_kernel(ik, k_ref)
        #        np.save('%s' % i, ik)
        #else:
        #    k = SOAPTools.center_kernel(k, k_ref)

    # Build structure kernels
    else:
        if args.kernel == 'gaussian':
            k = SOAPTools.build_sum_kernel_batch(projFiles, 
                    repSOAPs[:, 0:args.npca], structIdxs, 
                    kernel='gaussian', width=args.width, nc=args.npca)
        elif args.kernel == 'laplacian':
            k = SOAPTools.build_sum_kernel_batch(projFiles, 
                    repSOAPs[:, 0:args.npca], structIdxs,
                    kernel='laplacian', width=args.width, nc=args.npca)
        else: # linear kernel
            k = SOAPTools.build_sum_kernel_batch(projFiles, 
                    repSOAPs[:, 0:args.npca], structIdxs,
                    zeta=args.zeta, nc=args.npca)

        k = (k.T*3/nAtoms).T

        # Center structure kernel
        #k = SOAPTools.center_kernel(k, k_ref)

    # Perform projection with provided weights
    if args.env:
        SOAPTools.property_regression_oos_env(w, k, output=args.output)
    else:

        # Read mean
        f = open(args.mean, 'r')
        p_mean = float(f.readline().strip())
        f.close()

        # Scale property
        if args.p == 'volume':
            p /= nAtoms/3
            p -= p_mean
        
        # Energies
        else:
        
            # Convert to total energy
            p *= nAtoms/3

            # Remove mean binding energy
            p -= p_mean*nAtoms
        
            # Convert back to energy per Si
            p /= nAtoms/3

        SOAPTools.property_regression_oos(w, k, p, output=args.output)
