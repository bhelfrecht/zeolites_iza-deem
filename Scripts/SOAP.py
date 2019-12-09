#!/usr/bin/python

import os
import sys
import argparse
import random
import quippy as qp
import ase.io as aseIO
import numpy as np
import SOAPTools

parser = argparse.ArgumentParser()
parser.add_argument('-structure', type=str, default=None, 
        help='File containing structures')
parser.add_argument('-n', type=int, default=8, 
        help='Number of radial basis functions')
parser.add_argument('-l', type=int, default=6, 
        help='Number of angular basis functions')
parser.add_argument('-c', type=float, default=5.0, 
        help='Radial cutoff')
parser.add_argument('-cw', type=float, default=0.5, 
        help='Cutoff transition width')
parser.add_argument('-g', type=float, default=0.5, 
        help='Atomic Gaussian variance (width)')
parser.add_argument('-cwt', type=float, default=1.0, 
        help='Center atom weight')
parser.add_argument('-nonorm', action='store_true', 
        help='Normalisation True/False')
parser.add_argument('-z', type=int, nargs='+', default=None, 
        help='Space separated atomic numbers of environment species')
parser.add_argument('-Z', type=int, nargs='+', default=None, 
        help='Space separated atomic numbers of center species')
parser.add_argument('-cs', type=float, default=1.0, 
        help='Cutoff scale')
parser.add_argument('-cr', type=float, default=1.0, 
        help='Cutoff rate')
parser.add_argument('-dexp', type=int, default=0, 
        help='Cutoff exponent')
parser.add_argument('-idxs', type=str, default=None, 
        help='File with FPS indices')
parser.add_argument('-batchsize', type=int, default=0, 
        help='Number of structures in a batch')
parser.add_argument('-prefix', type=str, default='SOAPs', 
        help='Prefix for SOAP file names')
parser.add_argument('-output', type=str, default='.',
        help='Directory where output files should be saved')

args = parser.parse_args()

# SOAP normalization
if args.nonorm is True:
    nrm = 'F'
else:
    nrm = 'T'

# SOAP vector components to keep are stored in idxs
# Should just be 1D array, since want to select components
# based on random sample of structures in a single batch
if args.idxs is not None:
    idxs = np.loadtxt(args.idxs, dtype=np.int, unpack=True)
    if idxs.ndim > 1:
        idxs = idxs[0]
else:
    idxs = None

# Build input string to SOAP
nZ = len(args.Z)
Z = [str(i) for i in args.Z]
centersStr = str('{%s}' % (' '.join(Z)))

nz = len(args.z)
z = [str(i) for i in args.z]
envStr = str('{%s}' % (' '.join(z)))

soapStr = str('soap central_reference_all_species=F central_weight=%f ' \
              'covariance_sigma0=0.0 atom_sigma=%f cutoff=%f ' \
              'cutoff_transition_width=%f n_max=%d l_max=%d normalise=%s ' \
              'cutoff_scale=%f cutoff_rate=%f cutoff_dexp=%d ' \
              'n_Z=%d Z=%s n_species=%d species_Z=%s' % (args.cwt, args.g, 
                  args.c, args.cw, args.n, args.l, nrm, args.cs, args.cr, 
                  args.dexp, nZ, centersStr, nz, envStr))

# Set the descriptor
d = qp.descriptors.Descriptor(soapStr)
#al = qp.AtomsReader(args.structure)
al = aseIO.read(args.structure, index=':')

# Compute SOAP vectors for the input structure
SOAPTools.compute_SOAPs(al, d, idxs=idxs, 
        batchSize=args.batchsize, prefix=args.prefix, output=args.output)
