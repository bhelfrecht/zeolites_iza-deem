#!/usr/bin/env python

import os
import sys
import glob
import shlex
import subprocess
import argparse

def run_gulp(gulp_input, gulp_output, gulp_log):
    """
        Wrapper for running GULP calculations

        ---Arguments---
        gulp_input: GULP input file
        gulp_output: GULP output file
        gulp_log: GULP log file (for stderr)
    """

    gulp_in = open(gulp_input, 'r')
    gulp_out = open(gulp_output, 'w')
    gulp_log = open(gulp_log, 'w')

    # Run GULP
    gulp = subprocess.run('gulp', stdin=gulp_in, stdout=gulp_out, 
            stderr=gulp_log)

    gulp_in.close()
    gulp_out.close()
    gulp_log.close()

def cif2gulp(cif_name, gulp_name, library_file):
    """
        Read CIF files and turn them into GULP input files

        ---Arguments---
        input_name: name of CIF file
        output_name: name of GULP input file
        library_file: name of the library file
    """
    # TODO: make more compatible with COD structures

    basename = os.path.basename(cif_name)
    
    # Initialize the containers for storing the CIF data
    cif_data = {}
    atoms_data = {}
    atoms_data_order = []

    # Flag for centrosymmetry
    centrosymmetric = False
    
    # These are the keys from the CIF that we need for GULP
    cif_data_keys = (
            '_cell_length_a', 
            '_cell_length_b', 
            '_cell_length_c',
            '_cell_angle_alpha', 
            '_cell_angle_beta', 
            '_cell_angle_gamma',
            '_symmetry_Int_Tables_number', 
            '_symmetry_group_IT_number',
            '_symmetry_space_group_name_H-M', 
            '_space_group.IT_coordinate_system_code')

    # Multi-origin spacegroups
    multi_origin_IT = [
            48, 50, 59, 68, 
            70, 85, 86, 88, 
            125, 126, 129, 130, 
            133, 134, 137, 138, 
            141, 142, 201, 203, 
            222, 224, 227, 228
            ]
    multi_origin_HM = [
            'P N N N', 
            'P B A N', 
            'P M M N', 
            'C C C A', 
            'F D D D',
            'P 4/N', 
            'P 42/N', 
            'I 41/A', 
            'P 4/N B M', 
            'P 4/N N C', 
            'P 4/N M M',
            'P 4/N C C', 
            'P 4/N B C', 
            'P 4/N N M', 
            'P 4/N M C', 
            'P 4/N C M', 
            'I 41/A M D', 
            'I 41/A C D',
            'P N -3', 'P N 3',
            'F D -3', 'F D 3',
            'P N -3 N', 'P N 3 N',
            'P N -3 M', 'P N 3 M', 
            'F D -3 M', 'F D 3 M',
            'F D -3 C', 'F D 3 C'
            ]

    # These are just for reference
    origin_shifts = [
            [-1/4, -1/4, -1/4], 
            [-1/4, -1/4, 0], 
            [-1/4, -1/4, 0],
            [0, -1/4, -1/4], 
            [1/8, 1/8, 1/8], 
            [1/4, -1/4, 0], 
            [1/4, 1/4, 1/4],
            [0, 1/4, 1/8], 
            [1/4, 1/4, 0], 
            [1/4, 1/4, 1/4], 
            [1/4, -1/4, 0],
            [1/4, -1/4, 0], 
            [1/4, -1/4, 1/4], 
            [1/4, -1/4, 1/4], 
            [1/4, -1/4, 1/4],
            [1/4, -1/4, 1/4], 
            [0, -1/4, 1/8], 
            [0, -1/4, 1/8], 
            [1/4, 1/4, 1/4],
            [1/8, 1/8, 1/8], 
            [1/4, 1/4, 1/4], 
            [1/4, 1/4, 1/4], 
            [1/8, 1/8, 1/8], 
            [3/8, 3/8, 3/8]
            ]
    
    # Read the CIF file line-by-line
    with open(cif_name, 'r') as f:
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

            # Check for the symmetry operation (-x,-y,-z)
            # that indicates we must use the 2nd origin choice
            elif line.startswith('-x,-y,-z'):
                centrosymmetric = True
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
    g = open(gulp_name, 'w')
    
    # Optimization options
    g.write('opti conv shell\n')
    
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
    # we have available in the priority order H-M > IT > Int_Tables
    g.write('space\n')
    for label in ['_symmetry_space_group_name_H-M','_symmetry_group_IT_number',
            '_symmetry_Int_Tables_number']:
    
        # If we must use the H-M symbol, make sure it is capitalized
        if label in cif_data and label == '_symmetry_space_group_name_H-M':
            spacegroup = cif_data[label].upper()
            if spacegroup in multi_origin_HM and centrosymmetric:
                origin = 2
            else:
                origin = 1
            break
        elif label in cif_data:
            spacegroup = cif_data[label]
            if spacegroup in multi_origin_IT and centrosymmetric:
                origin = 2
            else:
                origin = 1
            break
    g.write(f'{spacegroup}\n')

    # Check for origin specification, overwriting automatic determination
    if '_space_group.IT_coordinate_system_code' in cif_data:
        origin = cif_data['_space_group.IT_coordinate_system_code']

    # Write origin
    g.write('origin\n')
    g.write(f'{origin}\n')

    # Species info
    g.write('species\n')
    g.write('Si core Si\n')
    g.write('O core O_O2-\n')
    g.write('O shel O_O2-\n')
    
    # Potential file
    g.write(f'library {library_file}\n')
    
    # Output files
    g.write(f'output xyz {gulp_name[:-3]}_opt\n')
    g.write(f'output cif {gulp_name[:-3]}_opt')
    g.close()

if __name__ == '__main__':

    # Parse command line arguments if called as script
    parser = argparse.ArgumentParser()
    parser.add_argument('cif', type=str,
            help='Directory containing the CIF files')
    parser.add_argument('gulp', type=str,
            help='Directory to store the GULP calculations')
    parser.add_argument('-l', '--library', type=str, default='.',
            help='Path to the library file')
    args = parser.parse_args()

    # Prepare for GULP run
    # We take the input dir as a command line option so we can
    # run different chunks of the dataset in parallel
    run_dir = os.getcwd()
    cif_dir = args.cif
    gulp_dir = args.gulp
    library_file = args.library
    cif_files = glob.glob(f'{cif_dir}/*.cif')

    if not os.path.exists(gulp_dir):
        os.makedirs(gulp_dir)
    
    for cif_file in cif_files:
    
        # Set up directories for GULP inputs and outputs
        basename = os.path.splitext(os.path.basename(cif_file))[0]
        gulp_run_dir = f'{gulp_dir}/{basename}'

        if not os.path.exists(gulp_run_dir):
            os.makedirs(gulp_run_dir)

        os.chdir(gulp_run_dir)
        gulp_input = f'{basename}.in'
        gulp_output = f'{basename}.out'
        gulp_log = f'{basename}.log'

        # Make GULP input files
        cif2gulp(os.path.relpath(cif_file, start=gulp_run_dir), gulp_input,
                os.path.relpath(library_file, start=gulp_run_dir))
    
        # Run GULP
        run_gulp(gulp_input, gulp_output, gulp_log) 
        os.chdir(run_dir)
