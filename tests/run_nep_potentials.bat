cd nep_potentials\carbon
..\..\..\src\gpumd < input.txt
fc thermo.out thermo1.out
del thermo.out
cd ..\..


