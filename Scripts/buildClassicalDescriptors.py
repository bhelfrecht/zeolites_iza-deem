#!/usr/bin/python

import os
import sys
import numpy as np
import quippy as qp
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-input', type=str, default=None, help='Input xyz file')
parser.add_argument('-output', type=str, default=None, help='Output filename')
parser.add_argument('-p', type=str, default='angles', help='Property name')

args = parser.parse_args()

# Open structure file
al = qp.AtomsReader(args.input, cache_mem_limit=1000)

# Create output file
g = open(args.output, 'w')

# Extract properties for each atom
for i, at in enumerate(al):

    # Parse atom properties for the structure
    P = np.asarray(at.properties[args.p].T)

    # Sort the vector for angles and distances
    # and write to output
    if args.p != 'rings':
        P.sort(axis=1)
        np.savetxt(g, np.flip(P, axis=1))
    else:
        np.savetxt(g, P[:, 1::2])

    sys.stdout.write('Frame: %d\r' % (i+1))

sys.stdout.write('\n')
g.close()
