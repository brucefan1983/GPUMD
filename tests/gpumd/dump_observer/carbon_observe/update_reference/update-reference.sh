
# First use the first potential
rm dump.xyz
sed -i 's@potential.*@potential        ../C_2022_NEP3.txt@g' run.in
~/repos/GPUMD/src/gpumd
cp dump.xyz ../reference_observer0.xyz

# Then use the other potential
# to predict energies, forces and virials
# and write them to the second reference.
source ~/venvs/organic/bin/activate
python predict_with_second_potential.py
mv predicted.xyz ../reference_observer1.xyz

# Clean up
rm dump.xyz neighbor.out
