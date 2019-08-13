#!/usr/bin/python

import os
import sys
import argparse
import quippy as qp
import numpy as np
import SOAPTools

parser = argparse.ArgumentParser()
parser.add_argument('-docf', action='store_true', help='Compute correlation factor')
parser.add_argument('-doscore', action='store_true', help='Compute DEEM structure score')
parser.add_argument('-structure', type=str, default=None, help='File containing structures')
parser.add_argument('-soap', type=str, 'SOAPFiles.dat',
                    help='File containing SOAP file filenames')
parser.add_argument('-deem', type=str, 'SOAPFiles.dat',
                    help='File containing SOAP file filenames')
parser.add_argument('-iza', type=str, 'SOAPFiles.dat',
                    help='File containing SOAP file filenames')
parser.add_argument('-pca', type=int, default=0, help='Number of PCA components')
parser.add_argument('-p', type=str, default='volume', choices=['volume', 'Energy_per_Si'], 
                    help='Property name for regression')
parser.add_argument('-Z', type=int, nargs='+', default=None, 
                    help='Space separated atomic numbers of center species')

args = parser.parse_args()

### COMPUTE CORRELATION FACTOR ###
if args.docf is True:
    f = open(args.soap, 'r')
    inputFiles = f.readlines()
    inputFiles = [i.strip() for i in inputFiles]
    f.close()
    al = qp.AtomsReader(args.structure)
    SOAPTools.correlation_factor(inputFiles, al, args.Z, args.pca)

### COMPUTE STRUCTURE SCORE ###
if args.doscore is True:
    f = open(args.deem, 'r')
    g = open(args.iza, 'r')
    DEEMFiles = f.readlines()
    DEEMFiles = [i.strip() for i in DEEMFiles]
    IZAFiles = g.readlines()
    IZAFiles = [i.strip() for i in IZAFiles]
    f.close()
    g.close()
    al = qp.AtomsReader(args.structure)
    SOAPTools.DEEM_score(DEEMFiles, IZAFiles, al, args.Z, nc=args.pca, propName=args.p, m='euclidean')
