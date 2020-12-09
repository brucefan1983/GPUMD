clear; close all;

load a.txt; % lattice constant (Angstrom)
N=size(a,1); % number of configurations for one deformation type

% get the outputs from GPUGA:

load energy.out;
load virial.out;
load force.out;


% normalized by the number of atoms
num_atoms=512; % I used 64 atoms in all the deformed configurations
energy=energy/num_atoms; % eV/atom
virial=virial/num_atoms; % eV/atom

% exclude a few polytypes in the figure
M=0; % number of configurations for some polytypes
energy=energy(M+1:end,:);
virial_xx=virial(0*end/6+M+1:1*end/6,:);
virial_yy=virial(1*end/6+M+1:2*end/6,:);
virial_zz=virial(2*end/6+M+1:3*end/6,:);
virial_xy=virial(3*end/6+M+1:4*end/6,:);
virial_yz=virial(4*end/6+M+1:5*end/6,:);
virial_zx=virial(5*end/6+M+1:6*end/6,:);

figure; % Figure 2 in the paper

% panel (a): force
subplot(1,3,1)
plot(force(1:end,4:6),force(1:end,1:3),'o');
hold on 
x=-2:0.5:2;
plot(x,x,'b-')
xlim([-2,2]);
ylim([-2,2]);
xlabel('Force (DFT) (eV/\AA)','fontsize',12,'interpreter','latex');
ylabel('Force (Potential) (eV/\AA)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
title('(a)');
% panel (b): energy
subplot(1,3,2);
plot(a,energy(1:end/3,2),'d');
hold on;
plot(a,energy(1:end/3,1),'-','linewidth',2);
plot(a,energy(end/3+1:2*end/3,2),'s');
plot(a,energy(end/3+1:2*end/3,1),'--','linewidth',2);
plot(a,energy(2*end/3+1:3*end/3,2),'o');
plot(a,energy(2*end/3+1:3*end/3,1),'-.','linewidth',2);
axis tight
xlabel('$a$ (\AA)','fontsize',12,'interpreter','latex');
ylabel('Energy (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
legend('triaxial-DFT','triaxial-Potential','biaxial-DFT',...
   'biaxial-Potential','uniaxial-DFT','uniaxial-Potential');
title('(b)');

% panel (c): virial in a deformed direction
subplot(1,3,3);
plot(a,virial_xx(1:end/3,2),'d','linewidth',1);
hold on;
plot(a,virial_xx(1:end/3,1),'-','linewidth',2);
plot(a,virial_xx(end/3+1:2*end/3,2),'d','linewidth',1);
plot(a,virial_xx(end/3+1:2*end/3,1),'-','linewidth',2);
plot(a,virial_zz(2*end/3+1:end,2),'s','linewidth',1);
plot(a,virial_zz(2*end/3+1:end,1),'--','linewidth',2);
axis tight
xlabel('$a$ (\AA)','fontsize',12,'interpreter','latex');
ylabel('Virial (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
title('(c)');


