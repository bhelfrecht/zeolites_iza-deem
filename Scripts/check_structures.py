#!/usr/bin/env python

import os
import sys
import glob
from tqdm import tqdm
import numpy as np
from ase.io import read, write
from ase import Atoms
import spglib as spg

# NOTE: this is mostly robust, but some structures
# still might "look" different according
# to the cell errors but are actually the same,
# depending on how we set the symprec
symprec = 1.0E-2
xyz_dir = '../Raw_Data/DEEM_330k/XYZ'
gulp_dir = '../Raw_Data/GULP/DEEM_330k'
xyz_files = sorted(glob.glob(f'{xyz_dir}/8*-8*/*.xyz'))
geometry_file = f'{gulp_dir}/geometry_errors.dat'

g = open(geometry_file, 'w')
g.write('# ID | Cell Error | Positions Error')

for xdx, xyz_file in enumerate(tqdm(xyz_files)):
    basename = os.path.splitext(os.path.basename(xyz_file))[0].rstrip('.cif')
    structure_id = int(basename)

    xyz = read(xyz_file, format='extxyz')
    cif = read(f'{gulp_dir}/{basename}/{basename}_opt.cif', format='cif')

    # Reduce XYZ (database structure) to primitive cell
    xyz_tuple = (xyz.cell[:], xyz.get_scaled_positions(), 
            xyz.get_atomic_numbers())

    primitive_xyz_cell, primitive_xyz_positions, primitive_xyz_numbers = \
            spg.find_primitive(xyz_tuple, symprec=symprec)

    primitive_xyz = Atoms(cell=primitive_xyz_cell, 
            scaled_positions=primitive_xyz_positions, 
            numbers=primitive_xyz_numbers, pbc=True)

    d_xyz = primitive_xyz.get_all_distances(mic=True)
    eig_xyz = np.linalg.eigvalsh(d_xyz)
    #idxs_xyz = np.argsort(np.abs(eig_xyz))
    idxs_xyz = np.argsort(eig_xyz)[::-1]

    # Reduce CIF (optimized structure) to primitive cell
    # (the optimized structures are already primitive, 
    # but we need it in the same form as the XYZ)
    cif_tuple = (cif.cell[:], cif.get_scaled_positions(),
            cif.get_atomic_numbers())

    primitive_cif_cell, primitive_cif_positions, primitive_cif_numbers = \
            spg.find_primitive(cif_tuple, symprec=symprec)

    primitive_cif = Atoms(cell=primitive_cif_cell,
            scaled_positions=primitive_cif_positions,
            numbers=primitive_cif_numbers, pbc=True)

    d_cif = primitive_cif.get_all_distances(mic=True)
    eig_cif = np.linalg.eigvalsh(d_cif)
    #idxs_cif = np.argsort(np.abs(eig_cif))
    idxs_cif = np.argsort(eig_cif)[::-1]

    # Calculate errors in cell and positions
    cell_error = np.linalg.norm(primitive_cif.cell[:] - primitive_xyz.cell[:])
    positions_error = np.linalg.norm(eig_cif[idxs_cif] - eig_xyz[idxs_xyz]) / np.sqrt(len(eig_cif))
    g.write(f'\n{structure_id:7d}  {cell_error:15.8f}  {positions_error:15.8f}')

g.close()
