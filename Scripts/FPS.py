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
        help='File containing SOAP vectors')
parser.add_argument('-fps', type=int, default=0, 
        help='Number of SOAP components to select via FPS')
parser.add_argument('-qfps', type=float, default=0, 
        help='Cutoff for Quick FPS')
parser.add_argument('-nr', type=int, default=0, 
        help='Number of random indices to select')
parser.add_argument('-c', action='store_true', 
        help='Select on components')
parser.add_argument('-output', type=str, default='.',
        help='Directory where outputs should be saved')
parser.add_argument('-nobatch', action='store_true',
        help='Concatenate multiple input files and perform FPS')
parser.add_argument('-d', action='store_true',
        help='Write distance information for FPS')

args = parser.parse_args()

# Read input file containing list of files that contain SOAP vectors
f = open(args.soap, 'r')
inputFiles = f.readlines()
inputFiles = [i.strip() for i in inputFiles]

# Can only do batch processing on environments, not components
if len(inputFiles) > 1 and args.c:
    sys.exit('Cannot select components reliably with batched data')

# Concatenate the SOAP vectors from the separate files
subFPS = []
fileIDs = []
newIdxs = []

n = 0
nEnv = 0

# If we are not computing FPS in batches,
# read the SOAP vectors from the batches
# and concatenate into a single matrix
if args.nobatch is True:
    SOAP = []
    for idx, i in enumerate(inputFiles):
        sys.stdout.write('Concatenating SOAPs, batch %d...\n' % (idx+1))
        iSOAP = SOAPTools.read_SOAP(i)
        SOAP.append(iSOAP)

    SOAP = np.concatenate(SOAP)

    # Transpose SOAP vectors if selecting on components
    # Use SOAP vectors as-is if selecting on environments
    if args.c:
        SOAP = SOAP.T

    nEnv = SOAP.shape[0]

    # Do "quick FPS"
    if args.qfps > 0:
        idxs = SOAPTools.quick_FPS(SOAP.T, D=args.fps, cutoff=args.qfps)
        np.savetxt('%s/quickFPS.idxs' % args.output, newIdxs, fmt='%d')
    
    # Do FPS
    elif args.fps > 0:
        idxs, distances = SOAPTools.do_FPS(SOAP, D=args.fps)

        # Save distances, if requested
        if args.d:
            np.savetxt('%s/FPS.idxs' % args.output, 
                    np.column_stack((idxs, distances)), 
                    fmt='%d\t%.16f')
        else:
            np.savetxt('%s/FPS.idxs' % args.output, idxs, fmt='%d')
    
else:

    # Loop over the batches
    # and do FPS for each batch
    for idx, i in enumerate(inputFiles):
        sys.stdout.write('Reading SOAPs in batch %d...\n' % (idx+1))
        SOAP = SOAPTools.read_SOAP(i)
    
        # Transpose SOAP vectors if selecting on components
        # Use SOAP vectors as-is if selecting environments
        if args.c:
            SOAP = SOAP.T
        
        # Do "quick FPS"
        if args.qfps > 0:
            idxs = SOAPTools.quick_FPS(SOAP.T, D=args.fps, cutoff=args.qfps)
            newIdxs.append(idxs+nEnv)
            subFPS.append(SOAP[idxs])
        
        # Do FPS
        elif args.fps > 0:
            idxs, distances = SOAPTools.do_FPS(SOAP, D=args.fps)
            newIdxs.append(idxs+nEnv)
            subFPS.append(SOAP[idxs])
    
        nEnv += len(SOAP)

    # Take our FPS points for each batch,
    # and do another FPS on the concatenated subselection
    if len(subFPS) > 1:
        sys.stdout.write('Selecting FPS Points from subsample...\n')
        newIdxs = np.concatenate(newIdxs)
        subFPS = np.concatenate(subFPS)
    
        # Do FPS on the concatenated FPS points
        if args.qfps > 0:
            idxs = SOAPTools.quick_FPS(subFPS.T, D=args.fps, cutoff=args.qfps)
            np.savetxt('%s/quickFPS.idxs' % args.output, newIdxs[idxs], fmt='%d')
        elif args.fps > 0:
            idxs, distances = SOAPTools.do_FPS(subFPS, D=args.fps)

            # Save distances, if requested
            if args.d:
                np.savetxt('%s/FPS.idxs' % args.output, 
                        np.column_stack((newIdxs[idxs], distances)), 
                        fmt='%d\t%.16f')
            else:
                np.savetxt('%s/FPS.idxs' % args.output, newIdxs[idxs], fmt='%d')

    # If we only have one batch,
    # we don't need to subselect, just save and exit
    else:
        newIdxs = np.asarray(newIdxs).flatten()
        if args.qfps > 0:
            np.savetxt('%s/quickFPS.idxs' % args.output, newIdxs, fmt='%d')
        elif args.fps > 0:

            # Save distances, if requested
            if args.d:
                np.savetxt('%s/FPS.idxs' % args.output, 
                        np.column_stack((newIdxs, distances)), 
                        fmt='%d\t%.16f')
            else:
                np.savetxt('%s/FPS.idxs' % args.output, newIdxs, fmt='%d')

# Random selection
if args.nr > 0:
    randomIdxs = range(0, nEnv)
    np.random.shuffle(randomIdxs)
    randomIdxs = randomIdxs[0:args.nr]
    #randomIdxs.sort()
    np.savetxt('%s/random.idxs' % args.output, randomIdxs, fmt='%d')

