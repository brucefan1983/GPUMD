clear; close all; font_size=10;

load mvac.out; % from GPUMD
load dos.out; % from GPUMD
num_atoms=8640; % from xyz.in
N=200; % number of correlation steps (from run.in)
M=length(mvac)/N % number of independent functions

t=mvac(1:N,1); % correlation time
nu=dos(1:N,1)/pi/2; % nu = omega/2/pi
vac_x=mean(reshape(mvac(:,2),N,M),2);
vac_y=mean(reshape(mvac(:,3),N,M),2);
vac_z=mean(reshape(mvac(:,4),N,M),2);
dos_x=mean(reshape(dos(:,2),N,M),2);
dos_y=mean(reshape(dos(:,3),N,M),2);
dos_z=mean(reshape(dos(:,4),N,M),2);

% An important check is that total PDOS should be normalized to num_atoms*3
% So these numbers should be close to 1:
normalization_of_pdos_x=trapz(nu,dos_x)/num_atoms
normalization_of_pdos_y=trapz(nu,dos_y)/num_atoms
normalization_of_pdos_z=trapz(nu,dos_z)/num_atoms


figure
% decompose
subplot(2,2,1);
plot(t,vac_x/vac_x(1),'r-','linewidth',1);
hold on;
plot(t,vac_y/vac_y(1),'b--','linewidth',1);
plot(t,vac_z/vac_z(1),'g-.','linewidth',1);
xlabel('Correlation time (ps)','fontsize',font_size);
ylabel('VAC (Normalized)','fontsize',font_size);
xlim([0,1]);
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);
legend('x','y','z');
title('(a)');

subplot(2,2,2);
plot(nu,dos_x,'r-',  nu,dos_y,'b--', nu,dos_z,'g-.', 'linewidth',1);
xlabel('\nu (THz)','fontsize',font_size);
ylabel('PDOS (1/THz)','fontsize',font_size);
xlim([0,55]);
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);
legend('x','y','z');
title('(b)');

% average over x, y, and z
subplot(2,2,3);
plot(t,(vac_x+vac_y+vac_z)/(vac_x(1)+vac_y(1)+vac_z(1)),'k-','linewidth',1);
xlabel('Correlation time (ps)','fontsize',font_size);
ylabel('VAC (Normalized)','fontsize',font_size);
xlim([0,1]);
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);
title('(c)');

subplot(2,2,4);
plot(nu, (dos_x + dos_y + dos_z)/3 ,'k-', 'linewidth', 1);
xlabel('\nu (THz)','fontsize',font_size);
ylabel('PDOS (1/THz)','fontsize',font_size);
xlim([0,55]);
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);
title('(d)');

% heat capacity
dT=100; % temperature interval
NT=50; % number of temperature points
temperature=zeros(NT,1);
C_x_per_atom=zeros(NT,1); % in units of k_B
C_y_per_atom=zeros(NT,1); % in units of k_B
C_z_per_atom=zeros(NT,1); % in units of k_B
for n=1:NT
    temperature(n)=n*dT;
    k_B_times_T=1.38e-23*temperature(n); % in units of J
    h_times_nu=6.63e-34*nu*1.0e12; % in units of J
    x=h_times_nu/k_B_times_T; % a dimensionless variable
    modal_heat_capacity=x.^2.*exp(x)./(exp(x)-1).^2; % in units of k_B
    C_x_per_atom(n)=trapz(nu,dos_x.*modal_heat_capacity)/num_atoms;
    C_y_per_atom(n)=trapz(nu,dos_y.*modal_heat_capacity)/num_atoms;
    C_z_per_atom(n)=trapz(nu,dos_z.*modal_heat_capacity)/num_atoms;
end

figure;
plot(temperature,C_x_per_atom,'d-','linewidth',1);
hold on;
plot(temperature,C_y_per_atom,'s-','linewidth',1);
plot(temperature,C_z_per_atom,'o-','linewidth',1);
xlabel('Temperature (K)','fontsize',15,'interpreter','latex');
ylabel('Heat capacity ($k_{\rm B}$/atom)','fontsize',15,'interpreter','latex');
xlim([0,5100]);
ylim([0,1.1]);
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
legend('x','y','z');





