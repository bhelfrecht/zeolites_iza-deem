#!/usr/bin/env python
 
# Script to convert cif to xyz considering symmetry using Atomic Simulation Environment
# by Rocio Semino

import os
import sys
from ase import Atoms 
from ase.io import read, write
from tqdm import tqdm

################################################################################
input_dir = sys.argv[1]
output_dir = sys.argv[2]

for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith('.cif'):
        config = read(filename, format='cif')
        write(f'{output_dir}/{filename}.xyz', config)
################################################################################
