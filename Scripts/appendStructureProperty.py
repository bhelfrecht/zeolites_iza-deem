#!/usr/bin/python

import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str,
        help='File containing property data')
parser.add_argument('-xyz', type=str,
        help='xyz file to add structure data to')
parser.add_argument('-c', type=int, default=[0], nargs='+',
        help='Columns to extract data from')
parser.add_argument('-n', type=str, default=['P0'], nargs='+',
        help='Property name(s) in same order as columns')
args = parser.parse_args()

# Exit if we don't have names for all of the properties
if len(args.c) != len(args.n):
    sys.exit('Must provide same number of names and columns')

# Load property values
properties = np.loadtxt(args.f, usecols=args.c)
if len(properties.shape) == 1:
    properties = np.reshape(properties, (properties.size, 1))

# Get file name fragments
dname, fname = os.path.split(os.path.abspath(args.xyz))

# Prepare output
g = open(os.path.join(dname, '%s_SP.xyz' % fname.split('.')[0]), 'w')
s = 0

# Read the xyz file that we want to augment
with open(args.xyz, 'r') as f:
    for line in f:
        line_data = line.strip().split()

        # We have a new structure
        if line_data[0].startswith('Lattice'):

            # Append the properties
            for ndx, nn in enumerate(args.n):
                line_data.append('%s=%.6f' % (nn, properties[s, ndx]))

            # Write the title line for the structure
            g.write(' '.join(line_data))
            g.write('\n')
            s += 1

        else:
            g.write(line)

g.close()
