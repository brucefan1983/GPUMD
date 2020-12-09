close all
clc
load a.txt
a=[a;a;a];
N=8000;
NN=63;
nxyz=10;
lattice_constant=5.46863;
energy_results=zeros(NN*3,2);
virial_results=zeros(NN*3,2);
j=1;
pressure=1.602177e+2;

for i = 1:1:NN
    x=a(i);
    v=x*x*x*nxyz*nxyz*nxyz;
    load (['.\Si_xyz\',num2str(i),'\thermo.out'])
    energy=thermo(1,3);
    virial=thermo(1,4)+thermo(1,5)+thermo(1,6);
    energy_results(j,1)=x;
    energy_results(j,2)=energy/N;
    virial_results(j,1)=x;
    virial_results(j,2)=v*virial/pressure/N;
    j=j+1;
 
end

for i = NN+1:1:NN*2
    x=a(i);
    v=lattice_constant*x*x*nxyz*nxyz*nxyz;
    load (['.\Si_xyz\',num2str(i),'\thermo.out'])
    energy=thermo(1,3);
    virial=thermo(1,4)+thermo(1,5)+thermo(1,6);
    energy_results(j,1)=x;
    energy_results(j,2)=energy/N;
    virial_results(j,1)=x;
    virial_results(j,2)=v*virial/pressure/N;
    j=j+1;
 
end

for i = 2*NN+1:1:3*NN
    x=a(i);
    v=lattice_constant*lattice_constant*x*nxyz*nxyz*nxyz;
    load (['.\Si_xyz\',num2str(i),'\thermo.out'])
    energy=thermo(1,3);
    virial=thermo(1,4)+thermo(1,5)+thermo(1,6);
    energy_results(j,1)=x;
    energy_results(j,2)=energy/N;
    virial_results(j,1)=x;
    virial_results(j,2)=v*virial/pressure/N;
    j=j+1;
 
end

fid=fopen('.\energy_MD.txt','w');
fprintf(fid,' %11.5f %11.5f \r\n',energy_results');
fclose(fid);
fid=fopen('.\virial_MD.txt','w');
fprintf(fid,' %11.5f %11.5f \r\n',virial_results');
fclose(fid);







