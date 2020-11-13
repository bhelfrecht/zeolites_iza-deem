#!/usr/bin/env python

import os
import sys
import glob
import shlex
import subprocess

def cif2gulp(input_name, output_name):
    # TODO: make more compatible with COD structures

    basename = os.path.basename(input_name)
    
    # Initialize the containers for storing the CIF data
    cif_data = {}
    atoms_data = {}
    atoms_data_order = []
    
    # These are the keys from the CIF that we need for GULP
    cif_data_keys = ('_cell_length_a', '_cell_length_b', '_cell_length_c',
            '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma',
            '_symmetry_Int_Tables_number', '_symmetry_group_IT_number',
            '_symmetry_space_group_name_H-M')
    
    # These are the keys relating to atom positions
    #atoms_data_keys = ('_atom_site_label', 
    #        '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z')
    
    # Read the CIF file line-by-line
    with open(input_name, 'r') as f:
        for line in f:
    
            # Skip comment lines
            if line.startswith('#'):
                continue
    
            # If we come across a piece of CIF data we need,
            # store it
            if line.startswith(cif_data_keys):
                line_data = shlex.split(line)
                cif_data[line_data[0]] = line_data[-1]
                continue
    
            # If we come across the keys that describe atoms,
            # keep track of what order they are in
            elif line.startswith('_atom_site'):
                line_data = line.strip()
                atoms_data[line_data] = []
                atoms_data_order.append(line_data)
                continue
    
            # Once we have all of the atom keys,
            # read the atomic positions and save
            # them in the appropriate dictionary entry
            if len(atoms_data_order) > 0:
                line_data = line.strip().split()
    
                # Double-check that the line indeed contains an atom
                # and extract the information, otherwise skip the line
                if len(line_data) == len(atoms_data_order):
                    for key, data in zip(atoms_data_order, line_data):
                        atoms_data[key].append(data)
    
    # Prepare the GULP input file
    g = open(output_name, 'w')
    
    # Optimization options
    g.write('conp shell\n')
    
    # Title info
    g.write('title\n')
    g.write(f'GULP input based on {basename}\n')
    g.write('end\n')
    
    # Unit cell
    g.write('cell\n')
    for key in ['_cell_length_a', '_cell_length_b', '_cell_length_c',
            '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']:
        g.write(f'\t{cif_data[key]}')
    g.write('\n')
    
    # Check for atom symbol or label
    if '_atom_site_type_symbol' in atoms_data:
        symbol_list = atoms_data['_atom_site_type_symbol']
    elif '_atom_site_label' in atoms_data:
        symbol_list = atoms_data['_atom_site_label']
    
    # Atom positions
    g.write('frac\n')
    for symbol, x, y, z in zip(symbol_list, atoms_data['_atom_site_fract_x'], 
            atoms_data['_atom_site_fract_y'], atoms_data['_atom_site_fract_z']):
        g.write(f'{symbol}\tcore\t{x}\t{y}\t{z}\n')
        if symbol == 'O':
            g.write(f'{symbol}\tshel\t{x}\t{y}\t{z}\n')
    
    # Write the spacegroup, using the format that
    # we have available in the priority order IT > Int_Tables > H-M
    g.write('space\n')
    for label in ['_symmetry_group_IT_number', '_symmetry_Int_Tables_number',
            '_symmetry_space_group_name_H-M']:
    
        # If we must use the H-M symbol, make sure it is capitalized
        if label in cif_data and label == '_symmetry_space_group_name_H-M':
            spacegroup = cif_data[label].upper()
            break
        elif label in cif_data:
            spacegroup = cif_data[label]
            break
    g.write(f'{spacegroup}\n')
    
    # Species info
    g.write('species\n')
    g.write('Si core Si\n')
    g.write('O core O_O2-\n')
    g.write('O shel O_O2-\n')
    
    # Potential file
    g.write('library DEEM2GULP.d/catlow_mod.lib\n')
    
    # Output files
    g.write(f'output xyz {output_name[:-4]}_OPT\n')
    g.write(f'output cif {output_name[:-4]}_OPT')
    g.close()

# Prepare for GULP run
output_dir = '../Raw_Data/GULP/DEEM_330k'

# We take the input dir as a command line option so we can
# run different chunks of the dataset in parallel
input_dir = sys.argv[1]
cif_files = glob.glob(f'{input_dir}/*.cif')
#cif_files = glob.glob('../Raw_Data/DEEM_330k/CIF/*.cif')

for cif_file in cif_files:

    # Set up directories for GULP inputs and outputs
    basename = os.path.splitext(os.path.basename(cif_file))[0]
    gulp_dir = f'{output_dir}/{basename}'
    if not os.path.exists(gulp_dir):
        os.makedirs(gulp_dir)
    gulp_file = f'{gulp_dir}/{basename}.in'

    # Make GULP input files
    cif2gulp(cif_file, gulp_file)

    # Run GULP
    # TODO: if we want to run in parallel with different
    # bash instances, we should change directory
    # before running GULP, but then we have to change
    # the library path in the GULP input
    gulp_in = open(gulp_file, 'r')
    gulp_out = open(f'{gulp_dir}/{basename}.out', 'w')

    subprocess.run('gulp', stdin=gulp_in, stdout=gulp_out)
