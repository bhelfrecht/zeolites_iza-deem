#!/usr/bin/python

import os
import sys
import argparse
import random
import quippy as qp
import numpy as np
import SOAPTools

parser = argparse.ArgumentParser()
parser.add_argument('-dopca', action='store_true', 
        help='Build the PCA')
parser.add_argument('-dotransform', action='store_true', 
        help='Transform with the PCA')
parser.add_argument('-doreconstruct', action='store_true', 
        help='Reconstruct data from PCA')
parser.add_argument('-soap', type=str, default='SOAPFiles.dat', 
        help='File containing SOAP file filenames')
parser.add_argument('-pca', type=int, default=0, 
        help='Number of PCA components')
parser.add_argument('-mean', type=str, default='mean.dat', 
        help='File with mean of data')
parser.add_argument('-w', type=str, default='eigenvectors.dat', 
        help='File with PCA eigenvectors')
parser.add_argument('-type', type=str, default='pca', choices=['raw', 'pca'],
        help="Reconstruct from raw data ('raw') or from PCA ('pca')")
parser.add_argument('-output', type=str, default='.',
        help='Directory where the output files should be saved')

args = parser.parse_args()

### BUILD PCA ###
if args.dopca is True:

    # Read SOAP vectors
    f = open(args.soap, 'r')
    inputFiles = f.readlines()
    inputFiles = [i.strip() for i in inputFiles]
    f.close()

    # Build incremental PCA
    SOAPTools.build_iPCA(inputFiles, args.pca, 
            batchSize=10000, output=args.output)

### TRANSFORM PCA ###
if args.dotransform is True:

    # Load eigenvectors and mean
    W = np.loadtxt(args.w)
    SOAPMean = np.loadtxt(args.mean)

    # Read SOAP vectors
    f = open(args.soap, 'r')
    inputFiles = f.readlines()
    inputFiles = [i.strip() for i in inputFiles]
    f.close()

    # Transform data according to PCA
    SOAPTools.transform_PCA(W, SOAPMean, inputFiles, output=args.output)

### RECONSTRUCT DATA ###
if args.doreconstruct is True:

    # Load eigenvectors and mean
    W = np.loadtxt(args.W)
    SOAPMean = np.loadtxt(args.mean)

    # Read SOAP vectors
    f = open(args.soap, 'r')
    inputFiles = f.readlines()
    inputFiles = [i.strip() for i in inputFiles]
    f.close()

    # Reconstruct data according to PCA
    if args.type == 'raw':
        SOAPTools.reconstruct_PCA(W, SOAPMean, inputFiles, 
                useRawData=True, output=args.output)
    else:
        SOAPTools.reconstruct_PCA(W, SOAPMean, inputFiles, 
                useRawData=False, output=args.output)

