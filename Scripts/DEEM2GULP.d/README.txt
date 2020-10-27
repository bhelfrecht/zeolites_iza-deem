Giovanni Pireddu, 2020

DEEM2GULP Workflow

==========TOOLKIT:
- ScanAll.sh:		The main script which runs everything else;
- cif2xyz.py:		A small script to convert a given .cif file into a .xyz file, using ase;
- GENERATE_INP.f90:	Converts the .xyz files into .inp files (GULP inputs);
- FINALIZE.f90:		Finalizes the .xyz files of the optimal structures, adding the lattice vectors in the comment lines;
- GET_ENERGIES.f90:	Extracts the values of each energy component from the log files.

==========USAGE:
- Replace lines 61 and 67 in ScanAll.sh with your GULP path
- Copy the desired .cif files in the same folder as the toolkit
- Run ScanAll.sh

==========OUTPUTS:
OPT_* files:		Optimal configurations, both in .cif and .xyz format
*.inp files:		Inputs for GULP
*.inp_DONE		Inputs for GULP, for which the optimization is already completed
log_GULP_* files: 	Log files containing all the information regarding the optimization of each structure. Basically the screen output of GULP.
Names_xyz.txt:		List of unoptimized .xyz structures
Names_cif.txt:		List of unoptimized .cif structures
Final_Confs.xyz:	OPT_*.xyz files collected in a single file
Energies.out:		Table of energy components for each structure
Rogues.out:		Always check that this is empty. Otherwise something went wrong with the optimizations. You will find the name of problematic structures here.
