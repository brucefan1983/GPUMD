clear; close all; font_size = 10; load shc.out; 

% parameters from compute_shc (check your run.in file)
Nc=250; % second parameter
Nw=1000; % fourth parameter
DT=19; % Temperature difference (K) from checking compute.out

% parameters from model (check your xyz.in file)
Ly=3*1.42*10; % length in transport direction for the chosen group(A)
Lx=1.42*sqrt(3)*100; % width (A)
Lz=3.35; % thickness of graphene (A)
V=Lx*Ly*Lz; % volume of the chosen group (A^3)

% calculated parameters
Nt=Nc*2-1;
time_in_ps=shc(1:Nt,1); % correlation time t (ps)
K=sum(shc(1:Nt,2:3),2)/Ly; % K(t) function (eV/ps)
nu=shc(Nt+1:end,1)/2/pi; % frequency (THz)
J=sum(shc(Nt+1:end,2:3),2); % spectral heat current (A*eV/ps/THz)
Gc=1.6e4*J/V/DT; % spectral thermal conductivity (W/m/K/THz)

% plot results
figure;
subplot(1,2,1);
plot(time_in_ps,K,'b-','linewidth',2);
set(gca,'fontsize', font_size);
xlabel('Correlation time (ps)','fontsize', font_size);
ylabel('K (eV/ps)','fontsize', font_size);
title('(a)');

subplot(1,2,2);
plot(nu, Gc, 'b-','linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('\omega/2\pi (THz)','fontsize',font_size);
ylabel('\kappa(\omega) (GW/m^2/K/THz)','fontsize',font_size);
xlim([0,53]);
set(gca,'ticklength',get(gca,'ticklength')*3,'xtick',0:10:50);
title('(b)');

save('Gc','Gc');
sum(Gc)*(nu(2)-nu(1))