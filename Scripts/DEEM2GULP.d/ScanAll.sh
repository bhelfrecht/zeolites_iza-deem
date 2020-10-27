## Giovanni Pireddu, 2019

#!/bin/bash

# Set the working directory
workdir='../../Raw_Data/GULP/DEEM_330k/TEST'

# Set the script directory
scriptdir=$PWD

# Copy over the catlow library
cp catlow_mod.lib "$workdir"

# Convert cif into .xyz, one by one.
cd "$workdir"

rm -f log*
rm -f OPT*
rm -f *.xyz
rm -f *.out
rm -f *.inp*
rm -f Names*
rm -f *.txt

echo 'Converting the .cif files into .xyz'

python3 "$scriptdir"/cif2xyz.py $PWD $PWD

echo 'Generating the input files for GULP'

for file in *.cif
do
    sed -i 's/(.*//g' $file
    filedir=${file::-4}
    mkdir -p $filedir
    mv $file $filedir

    # Need to iteratively build the list of names b/c
    # it is too long for ls
    echo $filedir/$file >> Names_cif.txt
done

for file in *.cif.xyz
do
    sed -i 's/Lattice="/Lattice=" /g'  $file
    sed -i 's/" Properties/ " Properties/g'  $file
    filedir=${file::-8}
    mv $file $filedir

    # Need to iteratively build the list of names b/c
    # it is too long for ls
    echo $filedir/$file >> Names_xyz.txt
done

gfortran "$scriptdir"/GENERATE_INP.f90 -o "$scriptdir"/GI.x
"$scriptdir"/GI.x

echo 'Optimizing structures'

for dir in */
do
    cd $dir
    file=${dir::-1}.inp
    mv ../$file .

    echo 'Working on '$file
    fwname=${file::-4}

    ## Change path to gulp here
    gulp < $file > log_GULP_$fwname.out

    if grep -Fq "Conditions for a minimum have not been satisfied" log_GULP_$fwname.out
    then
	    echo 'Optimal structure not achieved ------ Attempting constant V optimisation'
	    sed -i "s/opti conp/opti conv/" $file
        gulp < $file > log_GULP_$fwname.out
	    if grep -Fq "Conditions for a minimum have not been satisfied" log_GULP_$fwname.out
	    then
	        echo "Optimal structure not achieved ------ I will keep the result anyway"
	    fi
    fi
    #grep -c 'Si' 'OPT_'$fwname'.xyz' >> Energies.out
    #grep "kJ/(mole unit cells)" log_GULP_$fwname.out >> Energies.out
    mv $file $file'_DONE' 
    echo "$dir"OPT_$fwname.cif >> ../Names_OPT_cif.txt
    echo "$dir"OPT_$fwname.xyz >> ../Names_OPT_xyz.txt
    echo "$dir"log_GULP_$fwname.out >> ../Files.txt
    cd ..
done

# Finalize the optimized XYZ files
gfortran "$scriptdir"/FINALIZE.f90 -o "$scriptdir"/F.x
"$scriptdir"/F.x

#cat OPT*.xyz > Final_Confs.xyz
for dir in */
do
    id=${dir::-1}
    cat $id/OPT_$id.xyz >> Final_Confs.xyz
done

# Energies
gfortran "$scriptdir"/GET_ENERGIES.f90 -o "$scriptdir"/GE.x
"$scriptdir"/GE.x 

cd "$scriptdir"

echo 'Done :)'
