#!/usr/bin/python
import os
import sys
import argparse
import random
import quippy as qp
import numpy as np
import SOAPTools

parser = argparse.ArgumentParser()
parser.add_argument('-structure', type=str, default=None, 
        help='File containing structures')
parser.add_argument('-nt', type=int, default=0, 
        help='Number of total structures')
parser.add_argument('-nr', type=int, default=0, 
        help='Number of random structures to select')
parser.add_argument('-output', type=str, default='.',
        help='Directory where the output files should be located')

args = parser.parse_args()

# Create new .xyz file by concatenating random structures 
# from the input structure file
SOAPTools.get_random_structures(args.structure, args.nt, args.nr, args.output)
