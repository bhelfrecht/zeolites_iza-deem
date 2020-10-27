#!/usr/bin/env python
 
# Script to convert cif to xyz considering symmetry using Atomic Simulation Environment
# by Rocio Semino

import os
import sys
from ase import Atoms 
from ase.io import read, write
from tqdm import tqdm
import glob

################################################################################
input_dir = sys.argv[1]
output_dir = sys.argv[2]

cifs = glob.glob(f'{input_dir}/*.cif')

for filename in tqdm(cifs):
    config = read(filename, format='cif')
    output_filename = os.path.basename(filename)
    write(f'{output_dir}/{output_filename}.xyz', config)
################################################################################
