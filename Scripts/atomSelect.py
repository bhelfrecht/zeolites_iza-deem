#!/usr/bin/env python

import os
import sys
import numpy as np
import ase.io as aseIO
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('atoms', type=str, default=None,
        help='Atoms file')
parser.add_argument('xyz', type=str, default=None,
        help='XYZ file from which to extract structures')
parser.add_argument('indices', type=int, default=None, nargs='+',
        help='Indices of environments')
parser.add_argument('cutoff', type=float, default=None,
        help='Environment cutoff')

args = parser.parse_args()

f = open(args.atoms, 'r')
lines = f.readlines()
f.close()

index_lines = []
for i in args.indices:
    index_lines.append(lines[i])

index_lines = [i.strip().split() for i in index_lines]

env_indices = [int(i[0]) for i in index_lines]
atom_indices = [int(i[1]) for i in index_lines]
structure_indices = [int(i[7]) for i in index_lines]
structures = [i[-1].split('.')[0] for i in index_lines]

for sdx, s in enumerate(structure_indices):
    al = aseIO.read(args.xyz, index=s)
    n_atoms = len(al)
    abc = np.linalg.norm(al.cell, axis=0)
    abc_min = np.amin(abc)

    if abc_min < args.cutoff:
        n_rep = np.ceil(args.cutoff/abc_min)
        if n_rep % 2 == 0:
            n_rep += 1
    else:
        n_rep = 3

    al_rep = al.repeat((n_rep, n_rep, n_rep))
    n_cell = np.floor(n_rep*n_rep*n_rep/2)
    new_idx = n_cell*n_atoms+atom_indices[sdx]
    new_idx = int(new_idx)
    print "===== Environment", env_indices[sdx], "====="
    print "Index in %dx%dx%d replicated structure: %d" % (n_rep, n_rep,
            n_rep, new_idx)

    aseIO.write('%s_%dx%dx%d.xyz' % (structures[sdx], n_rep, n_rep, n_rep), 
            al_rep, format='extxyz')
