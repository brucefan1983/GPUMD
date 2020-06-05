cd gpumd\graphene_dos
del thermo.out dos.out mvac.out velocity.out
..\..\..\src\gpumd < input.txt
fc thermo.out thermo.out1
fc mvac.out mvac.out1
fc dos.out dos.out1
fc velocity.out velocity.out1
cd ..\..

cd gpumd\graphene_kappa_emd
del thermo.out hac.out
..\..\..\src\gpumd < input.txt
fc thermo.out thermo.out1
fc hac.out hac.out1
cd ..\..

cd gpumd\graphene_kappa_nemd
del thermo.out compute.out shc.out
..\..\..\src\gpumd < input.txt
fc thermo.out thermo.out1
fc compute.out compute.out1
fc shc.out shc.out1
cd ..\..

cd gpumd\graphene_kappa_hnemd
del thermo.out kappa.out shc.out
..\..\..\src\gpumd < input.txt
fc thermo.out thermo.out1
fc kappa.out kappa.out1
fc shc.out shc.out1
cd ..\..

cd phonon\silicon_dispersion
del D.out omega2.out
..\..\..\src\phonon < input.txt
fc D.out D.out1
fc omega2.out omega2.out1
cd ..\..

