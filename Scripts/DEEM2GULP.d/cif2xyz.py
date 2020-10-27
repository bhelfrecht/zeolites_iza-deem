#!/usr/bin/env python
 
# Script to convert cif to xyz considering symmetry using Atomic Simulation Environment
# by Rocio Semino

import os
import sh
from ase import Atoms 
from ase.io import read, write

################################################################################
for filename in os.listdir('.'):
	if filename.endswith(".cif"):
		config = Atoms(read(filename,format='cif'))
		write(filename+".xyz", config)
################################################################################
