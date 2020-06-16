cd gpumd\graphene
del compute.out
del kappa.out
del shc.out
del hac.out
del thermo.out
del dos.out
del mvac.out
del restart.out
del velocity.out
del movie.xyz
..\..\..\src\gpumd < input.txt
fc compute.out compute.out.ref
fc kappa.out kappa.out.ref
fc shc.out shc.out.ref
fc hac.out hac.out.ref
fc thermo.out thermo.out.ref
fc dos.out dos.out.ref
fc mvac.out mvac.out.ref
fc restart.out restart.out.ref
fc velocity.out velocity.out.ref
fc movie.xyz movie.xyz.ref
cd ..\..


