#!/usr/bin/python

import os
import sys
#import quippy as qp
import ase.io as aseIO
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-input', type=str, default=None,
        help='File to extract atoms from')
parser.add_argument('-output', type=str, default='atoms.dat',
        help='Output file')
parser.add_argument('-Z', type=int, default=[], nargs='+',
        help='Atomic numbers of atoms to list')
parser.add_argument('-sp', type=str, default=[], nargs='+',
        help='Structure properties to extract')
parser.add_argument('-ap', type=str, default=[], nargs='+',
        help='Atom properties to extract')

args = parser.parse_args()

# Load atoms list
#al = qp.AtomsReader(args.input)
al = aseIO.read(args.input, index=':')

# Open output file for writing
f = open(args.output, 'w')

# Central atoms
Z = set(args.Z)

# Initialize atom number and structure number
na = 0
ns = 0

# Loop over atoms list
for i, at in enumerate(al):

    # Parse the desired structure properties
    structureProperties = []
    for sp in args.sp:
        #structureProperties.append(at.params[sp])
        structureProperties.append(at.info[sp])
    v = np.linalg.det(at.cell)

    # Parse the desired atom properties (Fortran indexed)
    atomProperties = []
    for ap in args.ap:
        #atomProperties.append(at.properties[ap])
        atomProperties.append(at.arrays[ap])

    for j, aa in enumerate(at):
        line = []

        # Write out the atom numbers
        if aa.number in Z:
            f.write('%6d %6d %6d %2s %12.8f %12.8f %12.8f ' % (na, aa.index,
                aa.number, aa.symbol, 
                aa.position[0], aa.position[1], aa.position[2]))

            # Write out the atom properties
            for ap in atomProperties:
                aap = ap[j+1]
                f.write('%10s ' % str(aap))

            # Write out the structure properties
            f.write('%6d %12.8f ' % (ns, v))
            for sp in structureProperties:
                sp = str(sp)
                f.write('%10s ' % sp)
            f.write('\n')
            na += 1
    ns += 1

f.close()
