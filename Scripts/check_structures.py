#!/usr/bin/env python

import os
import sys
import glob
from tqdm import tqdm
import numpy as np
from ase.io import read
from ase import Atoms
from ase.cell import Cell
from ase.geometry import distance
import spglib as spg

xyz_dir = '../Raw_Data/DEEM_330k/XYZ'
gulp_dir = '../Raw_Data/GULP/DEEM_330k'
xyz_files = sorted(glob.glob(f'{xyz_dir}/830-833/*.xyz'))
geometry_file = f'{gulp_dir}/geometry_errors.dat'

g = open(geometry_file, 'w')
g.write('# ID | Cell Error | Positions Error')

for xdx, xyz_file in enumerate(tqdm(xyz_files[0:10])):
    basename = os.path.splitext(os.path.basename(xyz_file))[0].rstrip('.cif')
    structure_id = int(basename)

    xyz = read(xyz_file, format='extxyz')
    cif = read(f'{gulp_dir}/{basename}/{basename}_opt.cif', format='cif')

    xyz_tuple = (xyz.cell[:], xyz.get_scaled_positions(), 
            xyz.get_atomic_numbers())

    if xyz.info['spacegroup'] == 'P 1':
        primitive_xyz = xyz
        standard_primitive_xyz_cell, Q = primitive_xyz.cell.standard_form()
    else:
        primitive_xyz_cell, primitive_xyz_positions, primitive_xyz_numbers = \
                spg.find_primitive(xyz_tuple)

        primitive_xyz = Atoms(cell=primitive_xyz_cell, 
                scaled_positions=primitive_xyz_positions, 
                numbers=primitive_xyz_numbers)
        #print(cif.cell[:])
        #print(cif.cell.standard_form()[0][:])
        #print(primitive_xyz.cell.standard_form()[0][:])
        print(cif.cell.cellpar())
        print(primitive_xyz.cell.cellpar())

    #cell_error = np.linalg.norm(cif.cell.standard_form()[0][:] - standard_primitive_xyz_cell[:])
    #positions_error = distance(cif, primitive_xyz)
    #g.write(f'\n{structure_id:7d}  {cell_error:15.8f}  {positions_error:15.8f}')

g.close()
