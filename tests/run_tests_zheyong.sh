cd gpumd/graphene_dos
echo "#### graphene_dos"
../../../src/gpumd > /dev/null
diff -q thermo.out thermo1.out
diff -q mvac.out mvac1.out
diff -q dos.out dos1.out
diff -q velocity.out velocity1.out
rm thermo.out dos.out mvac.out velocity.out
cd ../..

cd gpumd/graphene_kappa_emd
echo "#### graphene_kappa_emd"
../../../src/gpumd > /dev/null
diff -q thermo.out thermo1.out
diff -q hac.out hac1.out
rm thermo.out hac.out
cd ../..

cd gpumd/graphene_kappa_nemd
echo "#### graphene_kappa_nemd"
../../../src/gpumd > /dev/null
diff -q thermo.out thermo1.out
diff -q compute.out compute1.out
diff -q shc.out shc1.out
rm thermo.out compute.out shc.out
cd ../..

cd gpumd/graphene_kappa_hnemd
echo "#### graphene_kappa_hnemd"
../../../src/gpumd > /dev/null
diff -q thermo.out thermo1.out
diff -q kappa.out kappa1.out
diff -q shc.out shc1.out
rm thermo.out kappa.out shc.out
cd ../..

cd gpumd/silicon_dispersion
echo "#### silicon_dispersion"
../../../src/gpumd > /dev/null
diff -q omega2.out omega21.out
rm D.out omega2.out
cd ../..

cd gpumd/carbon
echo "#### carbon"
../../../src/gpumd > /dev/null
diff -q thermo.out thermo1.out
diff -q neighbor.out neighbor1.out
rm thermo.out neighbor.out
cd ../..

cd gpumd/silicon_liquid
echo "#### silicon_liquid"
../../../src/gpumd > /dev/null
diff -q movie.xyz movie1.xyz
diff -q dump.xyz dump1.xyz
diff -q restart.xyz restart1.xyz
diff -q force.out force1.out
diff -q rdf.out rdf1.out
diff -q adf.out adf1.out
diff -q sdc.out sdc1.out
diff -q msd.out msd1.out
diff -q viscosity.out viscosity1.out
diff -q neighbor.out neighbor1.out
rm force.out adf.out rdf.out movie.xyz dump.xyz restart.xyz sdc.out msd.out viscosity.out neighbor.out
cd ../..
