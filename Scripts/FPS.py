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
for idx, i in enumerate(inputFiles):
    sys.stdout.write('Reading SOAPs in batch %d...\n' % (idx+1))
    if os.path.splitext(i)[1] == '.npy':
        SOAP = np.load(i)
    else:
        SOAP = np.loadtxt(i)

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
        idxs = SOAPTools.do_FPS(SOAP, D=args.fps)
        newIdxs.append(idxs+nEnv)
        subFPS.append(SOAP[idxs])

    nEnv += len(SOAP)

if len(subFPS) > 1:
    sys.stdout.write('Selecting FPS Points from subsample...\n')
    newIdxs = np.concatenate(newIdxs)
    subFPS = np.concatenate(subFPS)

    # Do FPS on the concatenated FPS points
    if args.qfps > 0:
        idxs = SOAPTools.quick_FPS(subFPS.T, D=args.fps, cutoff=args.qfps)
        np.savetxt('%s/quickFPS.idxs' % args.output, newIdxs[idxs], fmt='%d')
    elif args.fps > 0:
        idxs = SOAPTools.do_FPS(subFPS, D=args.fps)
        np.savetxt('%s/FPS.idxs' % args.output, newIdxs[idxs], fmt='%d')
else:
    newIdxs = np.asarray(newIdxs).flatten()
    if args.qfps > 0:
        np.savetxt('%s/quickFPS.idxs' % args.output, newIdxs, fmt='%d')
    elif args.fps > 0:
        np.savetxt('%s/FPS.idxs' % args.output, newIdxs, fmt='%d')

# Random selection
if args.nr > 0:
    randomIdxs = range(0, nEnv)
    np.random.shuffle(randomIdxs)
    randomIdxs = randomIdxs[0:args.nr]
    randomIdxs.sort()
    np.savetxt('%s/random.idxs' % args.output, randomIdxs, fmt='%d')

