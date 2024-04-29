# First run an average trajectory
rm dump.xyz
~/repos/GPUMD/src/gpumd

# Then predict with both potentials on the trajectory
source ~/venvs/organic/bin/activate
python predict_with_both_potentials.py
mv predicted0.xyz ../reference_observer0.xyz
mv predicted1.xyz ../reference_observer1.xyz

# Clean up
rm observer.xyz observer.out neighbor.out
