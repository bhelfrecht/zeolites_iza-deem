#!/usr/bin/python

import os
import sys
import numpy as np
import quippy as qp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-input', type=str, help='Input file to sort')
parser.add_argument('-output', type=str, help='Output file')
args = parser.parse_args()

# Read input file
al = qp.AtomsList(args.input)

# Sort the Atoms objects by filename,
# so that the order of the files
# matches the full DEEM xyz files
al.sort(attr='Filename')
out = qp.AtomsWriter(args.output)

# Write output
for i, at in enumerate(al):
    out.write(at)
out.close()



