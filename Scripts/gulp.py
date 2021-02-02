#!/usr/bin/env python

import os
import sys
import glob
import shlex
import subprocess
import argparse
from tqdm import tqdm
from ase import Atoms
from ase.io import read
from ase.geometry import distance
import numpy as np
import spglib as spg
from project_utils import removeprefix, removesuffix, get_basename

def find_deem_subdir(basename, base_dir):
    """
        Find the correct subdirectory of a
        Deem database structure based on its ID

        ---Arguments---
        basename: the structure ID
        base_dir: the directory where the structures are archived

        ---Returns---
        subdir: the directory within the base_dir where
            the desired structure (with ID basename)
            can be found
    """

    # Key for finding the subdirectories
    dir_codes = {
        '800-804': ['800', '801', '802', '803', '804'],
        '805-809': ['805', '806', '807', '808', '809'],
        '810-814': ['810', '811', '812', '813', '814'],
        '815-819': ['815', '816', '817', '818', '819'],
        '820-824': ['820', '821', '822', '823', '824'],
        '825-829': ['825', '826', '827', '828', '829'],
        '830-833': ['830', '831', '832', '833']
    }

    # Get first 3 digits of the ID and find the subdirectory
    basename_code = basename[0:3]
    for subdir, codes in dir_codes.items():
        if basename_code in codes:
            return subdir

def find(filename, base_dir):
    """
        Find a file by walking through subdirectories

        ---Arguments---
        filename: name of the file to find the path for
        base_dir: directory in which to begin the search

        ---Returns---
        filepath: filepath for the desired file
    """

    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)

def backup_run(files, backup_dir):
    """
        Backup a set of files

        ---Arguments---
        files: files to set aside in a separate directory
        backup_dir: new directory in which to move the files
    """

    os.mkdir(backup_dir)
    for f in files:
        os.rename(f, f'{backup_dir}/{f}')

def replace_input_line(old_file, new_file, old_line, new_line):
    """
        Replace a line of text in a file
        by making a copy

        ---Arguments---
        old_file: file in which to search for the matching line
        new_file: new file in which to write the file with the substitution
        old_line: text to match
        new_line: text to substitute in for old_line
    """

    g = open(new_file, 'w')
    with open(old_file, 'r') as f:
        for line in f:
            if old_line in line:
                g.write(line.replace(old_line, new_line))
            else:
                g.write(line)
    g.close()

def fix_gulp(gulp_dir, summary_file, logfile):
    """
        Find failed or unconverged GULP calculations
        and re-run

        ---Arguments---
        gulp_dir: directory containing the GULP calculations
        summary_file: file containing a summary of the GULP calculations
            from gulp_summary
        logfile: file in which to keep a log of failed/unconverged calculations
    """

    run_dir = os.getcwd()
    log = open(logfile, 'w')

    # Search the summary file
    with open(summary_file, 'r') as f:
        for line in f:
    
            if line.startswith('#'):
                continue
    
            # Get summary data: structure ID, and whether the
            # calculation failed or didn't converge
            line_data = line.strip().split()
            structure_id = line_data[0]
            failed_minimum = int(line_data[-2])
            failed = int(line_data[-1])
    
            # If a calculation fails or doesn't converge,
            # restart from scratch and do a constant
            # volume optimization
            if failed or failed_minimum:
                if failed:
                    log.write(f'GULP for structure {structure_id} failed. ')
                elif failed_minimum:
                    log.write(f'No minimum found for {structure_id}. ')
                
                log.write('Attempting constant volume calculation\n')
    
                os.chdir(f'{gulp_dir}/{structure_id}')
    
                # Back up the old calculation (assumes it was at constant pressure)
                failed_dir = 'CONP'
                failed_files = glob.glob(f'{structure_id}*')
                backup_run(failed_files, failed_dir)
    
                # Create new GULP inputs for constant volume calculation
                new_gulp_input = f'{structure_id}.in'
                new_gulp_output = f'{structure_id}.out'
                new_gulp_log = f'{structure_id}.log'
                replace_input_line(
                    f'{failed_dir}/{structure_id}.in', 
                    new_gulp_input, 'opti conp', 'opti conv'
                )

                # Run GULP
                run_gulp(new_gulp_input, new_gulp_output, new_gulp_log)
    
                os.chdir(run_dir)
    
    log.close()

def structure_summary(output, gulp_dir, gulp_glob, ref_dir, ref_ext,
        primitive=False, symprec=1.0E-2, wrapeps=1.0E-12):
    """
        Compute differences in cell parameters and atomic positions

        ---Arguments---
        output: file in which to write the geometry data
        gulp_dir: directory containing the GULP calculations
        ref_dir: directory containing the reference structures
        gulp_glob: glob pattern for the GULP files
        ref_ext: file extension the reference files
        primitive: whether to compute the geometry differences using
            the primitive unit cell. Returns position differences
            as the norm of the eigenvalue spectrum of the intra-structure
            distance matrix. Otherwise the Frobenius norm of inter-structure
            distances is returned according to ase.geometry.distance
        symprec: symmetry precision for determining primitive cells
        wrapeps: precision for wrapping structures after ASE loading
    """

    g = open(output, 'w')
    g.write('# ID | Cell Error | Positions Error')
    if primitive:
        g.write(' (Eigenvalues)')
    else:
        g.write(' (Distances)')

    # Get output structures (in cif) from the GULP calculations.
    # We have to use CIF, as it has the cell information
    gulp_files = sorted(glob.glob(f'{gulp_dir}/{gulp_glob}'))
    for gulp_file in tqdm(gulp_files):
        basename = removesuffix(get_basename(gulp_file), '_opt')

        # Find the reference structures: if we can't find it in the
        # reference directory, assume we are dealing with the
        # Deem frameworks and search the 8*-8* subdirectories
        ref_basename = f'{basename}.{ref_ext}'
        if ref_basename not in os.listdir(ref_dir):
            ref_subdir = find_deem_subdir(ref_basename, ref_dir)
            ref_file = f'{ref_dir}/{ref_subdir}/{ref_basename}'
        else:
            ref_file = f'{ref_dir}/{ref_basename}'

        gulp_cif = read(gulp_file)
        gulp_cif.wrap(eps=wrapeps)

        ref_xyz = read(ref_file)
        ref_xyz.wrap(eps=wrapeps)

        if primitive:

            # NOTE: this is mostly robust, but some structures
            # still might "look" different according
            # to the cell errors but are actually the same,
            # depending on how we set the symprec
            gulp_tuple = (
                gulp_cif.cell[:],
                gulp_cif.get_scaled_positions(),
                gulp_cif.get_atomic_numbers()
            )
            gulp_cell, gulp_pos, gulp_numbers = \
                spg.find_primitive(gulp_tuple, symprec=symprec)

            gulp = Atoms(
                cell=gulp_cell, 
                scaled_positions=gulp_pos, 
                numbers=gulp_numbers, pbc=True
            )

            d_gulp = gulp.get_all_distances(mic=True)
            eig_gulp = np.linalg.eigvalsh(d_gulp)
            #idxs_gulp = np.argsort(np.abs(eig_gulp))
            idxs_gulp = np.argsort(eig_gulp)[::-1]

            ref_tuple = (
                ref_xyz.cell[:],
                ref_xyz.get_scaled_positions(),
                ref_xyz.get_atomic_numbers()
            )
            ref_cell, ref_pos, ref_numbers = \
                spg.find_primitive(ref_tuple, symprec=symprec)

            ref = Atoms(
                cell=ref_cell,
                scaled_positions=ref_pos,
                numbers=ref_numbers, pbc=True
            )

            d_ref = ref.get_all_distances(mic=True)
            eig_ref = np.linalg.eigvalsh(d_ref)
            #idxs_ref = np.argsort(np.abs(eig_ref))
            idxs_ref = np.argsort(eig_ref)[::-1]

            cell_error = np.linalg.norm(ref.cell[:] - gulp.cell[:])
            pos_error = np.linalg.norm(eig_ref[idxs_ref] - eig_gulp[idxs_gulp]) \
                / np.sqrt(len(eig_ref))
            
        else:
            cell_error = np.linalg.norm(ref_xyz.cell[:] - gulp_cif.cell[:])
            pos_error = distance(ref_xyz, gulp_cif)

        g.write(f'\n{basename:7s}  {cell_error:15.8f}  {pos_error:15.8f}')

    g.close()

def gulp_summary(output, gulp_dir, gulp_glob, ref_ext=None, ref_dir=None):
    """
        Generate a summary of the GULP calculations

        ---Arguments---
        output: file in which to write the summary data
        gulp_dir: directory containing the GULP calculations
        gulp_glob: glob pattern for GULP files
        ref_dir: directory containing the reference structures
            (Deem database frameworks only)
    """

    # Energy conversion
    ev_to_kJmol = 96.485307

    # Find the GULP outputs
    gulp_files = sorted(glob.glob(f'{gulp_dir}/{gulp_glob}'))

    # If we already have a summary file, make a new one
    # (so that we can keep two separate summaries: after the original GULP
    # calculation, and after the "fix" for failed/unconverged calculations)
    if os.path.exists(output):
        new_name = '_fix'.join(os.path.splitext(output))
        g = open(new_name, 'w')
    else:
        g = open(output, 'w')
    
    # Generate header
    pre_header = '# ID |'
    post_header = 'GULP Energy | GULP Gnorm | Minimum Failed | GULP Failed'
    if ref_dir is not None:
        header = f'{pre_header} Reference Energy | {post_header}'
    else:
        header = f'{pre_header} {post_header}'

    g.write(header)

    for gulp_file in tqdm(gulp_files):
        gulp_energy = np.nan
        gulp_gnorm = np.nan
        gulp_minimum_failed = 0 
        gulp_failed = 0
    
        basename = get_basename(gulp_file)
        g.write(f'\n{basename:7s}')
    
        gulp_file = f'{gulp_dir}/{basename}/{basename}.out'
        gulp_error_file = f'{gulp_dir}/{basename}/{basename}.log'
    
        if os.path.getsize(gulp_error_file) > 0:
            gulp_failed = 1 
    
        # Extract the reference energy from the Deem database CIF
        if ref_dir is not None and ref_ext is not None:
            ref_subdir = find_deem_subdir(basename, ref_dir)
            ref_file = f'{ref_dir}/{ref_subdir}/{basename}.{ref_ext}'
            ref_energy = np.nan
            ref_target = 'GULP energy per Si atom'
            with open(ref_file, 'r') as ref:
                for line in ref:
                    if ref_target in line:
                        ref_energy = float(line.strip().split()[-2])
                        break
            g.write('  {ref_energy:15.8f}')
    
        # Extract the energies and Gnorms from the GULP calculation
        n = 0
        with open(gulp_file, 'r') as gulp:
            for line in gulp:
                if line.startswith('  Total number atoms/shells'):
                    n = int(line.strip().split()[-1]) / 5
                elif 'Conditions for a minimum have not been satisfied' in line:
                    gulp_minimum_failed = 1
                elif line.startswith('  Final energy') and n > 0:
                    gulp_energy = ev_to_kJmol * float(line.strip().split()[-2]) / n
                elif line.startswith('  Final Gnorm'):
                    gulp_gnorm = float(line.strip().split()[-1])
                    break
        
        g.write(
            f'  {gulp_energy:15.8f}  {gulp_gnorm:15.8f}  '
            f'{gulp_minimum_failed:1d}  {gulp_failed:1d}'
        )
    
    g.close()

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

def xyz2gulp(xyz_name, gulp_name, gulp_keywords, library_file):
    """
        Build a GULP input from an XYZ file
        (no symmetry)

        ---Arguments---
        xyz_name: name of the XYZ file on which to base the input
        gulp_name: name of GULP input file
        gulp_keywords: keyword options for GULP
        library_file: filepath to the GULP potential file
    """

    basename = os.path.basename(xyz_name)

    # Parse the XYZ
    symbols = []
    positions = []
    with open(xyz_name, 'r') as f:
        lines = f.readlines()
        header = lines.pop(1)
        n_Si = int(lines.pop(0))
        for line in lines:
            atom = line.strip().split()
            symbols.append(atom[0])
            positions.append('\t'.join(atom[1:]))

    # Extract cell information from the header
    header = shlex.split(header)
    for entry in header:
        if entry.startswith('Lattice'):
            lattice_str = removeprefix(entry, 'Lattice=')
            lattice = [[float(xyz) for xyz in lattice_str.split()[i:i+3]] \
                    for i in range(0, 9, 3)]

    # Prepare the GULP input file
    g = open(gulp_name, 'w')
    
    # Optimization options
    g.write(f'{gulp_keywords}\n')
    
    # Title info
    g.write('title\n')
    g.write(f'GULP input based on {basename}\n')
    g.write('end\n')
    
    # Unit cell
    g.write('vectors\n')
    for xyz in lattice:
        g.write(f'{xyz[0]:12.8f}\t{xyz[1]:12.8f}\t{xyz[2]:12.8f}\n')
    
    # Atom positions
    g.write('cartesian\n')
    for symbol, position in zip(symbols, positions):
        g.write(f'{symbol}\tcore\t{position}\n')
        if symbol == 'O':
            g.write(f'{symbol}\tshel\t{position}\n')
    
    # Write the spacegroup, using the format that
    # we have available in the priority order H-M > IT > Int_Tables
    #g.write('space\nP 1\n')

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

def cif2gulp(cif_name, gulp_name, gulp_keywords, library_file):
    """
        Read CIF files and turn them into GULP input files
        (with symmetry)

        ---Arguments---
        input_name: name of CIF file
        output_name: name of GULP input file
        gulp_keywords: input keyword(s) for GULP
        library_file: name of the library file
    """

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
    #g.write('opti conv shell kjmol\n')
    g.write(f'{gulp_keywords}\n')
    
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
    parser.add_argument('glob_pattern', type=str,
            help='Glob pattern for finding the input files')
    parser.add_argument('gulp_dir', type=str,
            help='Directory to store the GULP calculations')
    parser.add_argument('-lf', '--library_file', type=str, 
            default='../Raw_Data/GULP/catlow_mod.lib',
            help='Path to the library file')
    parser.add_argument('-kw', '--keywords', type=str, 
            default='opti conv shell',
            help='Keyword arguments for GULP')
    parser.add_argument('-idxf', '--index_file', type=str,
            default=None,
            help='Filename containing indices for subselecting structures')
    parser.add_argument('-ext', '--extension', type=str,
            default='xyz',
            help='Filename extension for searching files and building inputs')
    args = parser.parse_args()

    # Prepare for GULP run
    # We take the input dir as a command line option so we can
    # run different chunks of the dataset in parallel
    run_dir = os.getcwd()

    # Choose input conversion function based on extension
    if args.glob_pattern.endswith('cif'):
        frame2gulp = cif2gulp
    else:
        frame2gulp = xyz2gulp

    if args.index_file is not None:
        idxs = np.loadtxt(args.index_file, dtype=int)
        frame_files = sorted(glob.glob(args.glob_pattern))
        frame_files = [frame_files[i] for i in idxs]
    else:
        frame_files = glob.iglob(args.glob_pattern)

    os.makedirs(args.gulp_dir, exist_ok=True)
    
    for frame_file in frame_files:
    
        # Set up directories for GULP inputs and outputs
        # Extra strip for '.cif' because the Deem 330k XYZ are named *.cif.xyz
        basename = removesuffix(get_basename(frame_file), '.cif')
        gulp_run_dir = f'{args.gulp_dir}/{basename}'

        os.makedirs(gulp_run_dir, exist_ok=True)

        os.chdir(gulp_run_dir)
        gulp_input = f'{basename}.in'
        gulp_output = f'{basename}.out'
        gulp_log = f'{basename}.log'

        # Make GULP input files
        frame2gulp(
            os.path.relpath(frame_file, start=gulp_run_dir), 
            gulp_input, args.keywords, 
            os.path.relpath(args.library_file, start=gulp_run_dir)
        )
    
        # Run GULP
        run_gulp(gulp_input, gulp_output, gulp_log) 
        os.chdir(run_dir)
