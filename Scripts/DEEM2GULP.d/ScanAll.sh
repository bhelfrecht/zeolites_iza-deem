## Giovanni Pireddu, 2019

#!/bin/bash

rm -f log*
rm -f OPT*
rm -f *.xyz
rm -f *.out
rm -f *.inp*

### Convert cif into .xyz, one by one.

ls *.cif > Names_cif.txt

echo 'Converting the .cif files into .xyz'

mkdir Temp.d
   
for file in *.cif
do
    echo 'Working on' $file
    cp $file Temp.d
    cp cif2xyz.py Temp.d
    cd Temp.d
    python3 cif2xyz.py
    cp *.xyz ..
    rm *
    cd ..
done
   
rm -r Temp.d

ls *.xyz > Names_xyz.txt

######################################

echo 'Generating the input files for GULP'

for file in *.cif
do
    sed -i 's/(.*//g' $file
done

for file in *.cif.xyz
do
    sed -i 's/Lattice="/Lattice=" /g'  $file
    sed -i 's/" Properties/ " Properties/g'  $file
done

gfortran GENERATE_INP.f90 -o GI.x
./GI.x

echo 'Optimizing structures'

for file in *.inp
do
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
done

ls OPT*.xyz > Names_OPT_xyz.txt
ls OPT*.cif > Names_OPT_cif.txt

gfortran FINALIZE.f90 -o F.x
./F.x

cat OPT*.xyz > Final_Confs.xyz

#### Energies

gfortran GET_ENERGIES.f90 -o GE.x
./GE.x 

####################

echo 'Done :)'
