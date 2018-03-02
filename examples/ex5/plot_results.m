clear; close all; font_size=12;

load vac.out;
N=200; % number of correlation steps
M=length(vac)/N % number of independent functions

t=vac(1:N,1); % correlation time
nu=vac(1:N,8)/pi/2; % nu = omega/2/pi
vac_x=mean(reshape(vac(:,2),N,M),2);
vac_y=mean(reshape(vac(:,3),N,M),2);
vac_z=mean(reshape(vac(:,4),N,M),2);
dos_x=mean(reshape(vac(:,9),N,M),2);
dos_y=mean(reshape(vac(:,10),N,M),2);
dos_z=mean(reshape(vac(:,11),N,M),2);

figure
plot(t,(vac_x+vac_y+vac_z)/3,'-','linewidth',2);
xlabel('Correlation time (ps)','fontsize',font_size);
ylabel('VAC (A^2/ps^2)','fontsize',font_size);
xlim([0,0.8]);
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);

figure
plot(nu, (dos_x + dos_y + dos_z)/3 ,'-', 'linewidth', 2);
xlabel('\nu (THz)','fontsize',font_size);
ylabel('PDOS (1/THz)','fontsize',font_size);
xlim([0,40]);
ylim([0,0.14]);
set(gca,'fontsize',font_size);
set(gca,'xtick',0:5:40);
set(gca,'ticklength',get(gca,'ticklength')*2);

% An important check is that PDOS should be normalized to about 1/2
normalization_of_pdos_x=trapz(nu,dos_x)
normalization_of_pdos_y=trapz(nu,dos_y)
normalization_of_pdos_z=trapz(nu,dos_z)

