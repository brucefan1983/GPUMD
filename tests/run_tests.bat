cd gpumd\graphene_dos
..\..\..\src\gpumd < input.txt
fc thermo.out thermo.out1
fc mvac.out mvac.out1
fc dos.out dos.out1
fc velocity.out velocity.out1
del thermo.out dos.out mvac.out velocity.out
cd ..\..

cd gpumd\graphene_kappa_emd
..\..\..\src\gpumd < input.txt
fc thermo.out thermo.out1
fc hac.out hac.out1
del thermo.out hac.out
cd ..\..

cd gpumd\graphene_kappa_nemd
..\..\..\src\gpumd < input.txt
fc thermo.out thermo.out1
fc compute.out compute.out1
fc shc.out shc.out1
del thermo.out compute.out shc.out
cd ..\..

cd gpumd\graphene_kappa_hnemd
..\..\..\src\gpumd < input.txt
fc thermo.out thermo.out1
fc kappa.out kappa.out1
fc shc.out shc.out1
del thermo.out kappa.out shc.out
cd ..\..

cd phonon\silicon_dispersion
..\..\..\src\phonon < input.txt
fc D.out D.out1
fc omega2.out omega2.out1
del D.out omega2.out
cd ..\..

