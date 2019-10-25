#!/usr/bin/env python

import os
import sys
import numpy as np
import ase.io as aseIO
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('xyz', type=str, default=None,
        help='XYZ file from which to extract structures')
parser.add_argument('indices', type=int, default=None, nargs='+',
        help='Indices of structures')

args = parser.parse_args()

for sdx, s in enumerate(args.indices):
    al = aseIO.read(args.xyz, index=s)
    al = al.repeat((3, 3, 3))
    aseIO.write('%d_3x3x3.xyz' % s, al, format='extxyz')
