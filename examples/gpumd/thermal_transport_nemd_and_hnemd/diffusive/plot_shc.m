clear; close all; font_size = 10; load shc.out; 

% parameters from compute_shc (check your run.in file)
Nc=250; % second parameter
Nw=1000; % fourth parameter
T=300; % Temperature (K) 
Fe=1.0e-5; % driving force parameter (1/A)

% parameters from model (check your xyz.in file)
Ly=3*1.42*10; % length in transport direction for the chosen group (A)
Lx=1.42*sqrt(3)*100; % width (A)
Lz=3.35; % thickness of graphene (A)
V=Lx*Ly*Lz; % volume of the chosen group (A^3)

% calculated parameters and results
% Ref. [1]:  Z. Fan et al., PRB 99, 064308 (2019)
Nt=Nc*2-1;
time_in_ps=shc(1:Nt,1); % correlation time t (ps)
K=sum(shc(1:Nt,2:3),2)/Ly; % Eq. (18) in Ref. [1] divided by length (eV/ps)
nu=shc(Nt+1:end,1)/2/pi; % frequency (THz)
J=sum(shc(Nt+1:end,2:3),2); % left part of Eq. (20) in Ref. [1] (A*eV/ps/THz)
kappa=1.6e3*J/V/T/Fe; % left part of Eq. (21) in Ref. [1] (W/m/K/THz)
load ../ballistic/Gc;
lambda_i=kappa./Gc; % Eq. (22) in Ref. [1] (nm)
len=10.^(1:0.1:6); % consider length from 10 nm to 1 mm
for l=1:length(len)
    tmp=kappa./(1+lambda_i/len(l)); % Eq. (24) in Ref. [1]
    k_L(l)=sum(tmp)*(nu(2)-nu(1)); % Eq. (23) in Ref. [1]
end

% plot results
figure;
subplot(2,2,1);
plot(time_in_ps,K,'b-','linewidth',2);
set(gca,'fontsize', font_size);
xlabel('Correlation time (ps)','fontsize', font_size);
ylabel('K (eV/ps)','fontsize', font_size);
title('(a)');

subplot(2,2,2);
plot(nu, kappa, 'b-','linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('\omega/2\pi (THz)','fontsize',font_size);
ylabel('\kappa(\omega) (W/m/K/THz)','fontsize',font_size);
ylim([0,200]);
xlim([0,53]);
set(gca,'ticklength',get(gca,'ticklength')*3,'xtick',0:10:50);
title('(b)');

subplot(2,2,3);
plot(nu,lambda_i,'b-','linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('\omega/2\pi (THz)','fontsize',font_size);
ylabel('\lambda(\omega) (nm)','fontsize',font_size);
ylim([0,6000]);
xlim([0,53]);
set(gca,'ticklength',get(gca,'ticklength')*3,'xtick',0:10:50);
title('(c)');

subplot(2,2,4);
semilogx(len/1000,k_L,'b-','linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('L (\mum)','fontsize',font_size);
ylabel('\kappa (W/mK)','fontsize',font_size);
ylim([0,3200]);
xlim([1.0e-2,10^3]);
set(gca,'ticklength',get(gca,'ticklength')*3,'xtick',10.^(-2:1:6));
title('(d)');
