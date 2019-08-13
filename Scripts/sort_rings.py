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

al = qp.AtomsList(args.input)
al.sort(attr='Filename')
out = qp.AtomsWriter(args.output)

for i, at in enumerate(al):
    out.write(at)
out.close()



