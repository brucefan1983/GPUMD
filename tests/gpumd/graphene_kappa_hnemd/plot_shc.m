clear; close all; font_size = 10;
load shc.out;

% input parameters for SHC
dt=2; %fs
L=3*1.42*10; % A

% calculated parameters
Nc=size(shc,1);
dt_in_ps = dt/1000;  % ps
time_in_ps = (0:Nc-1)*dt_in_ps;
nu=(0.01:0.01:60);   % THz
k=sum(shc,2)*1000/10.18/L; %eV/ps
k=k.';

figure;
subplot(2,2,1);
plot(time_in_ps,k,'b-','linewidth',2);
set(gca,'fontsize', font_size);
xlabel('Correlation time (ps)','fontsize', font_size);
ylabel('K (eV/ps)','fontsize', font_size);
xlim([0, 0.5]);
title('(a)');

% use K(-t) = K(t) symmetry
k=k.*[1,2*ones(1,Nc-1)];

% Hann window
k=k.*(cos(pi*(0:Nc-1)/Nc)+1)*0.5;

% the Fourier transform
q=zeros(length(nu),1);
% use discrete cosine transform
for n=1:length(nu)
   q(n)=2*dt_in_ps*sum(k.*cos(2*pi*nu(n)*(0:Nc-1)*dt_in_ps));
end

Fe=0.00001; %1/A
T=300; 
A=0.142*sqrt(3)*100*0.335; % nm^2
kappa=16*q/A/T/Fe;

load ../ballistic/Gc;
lambda_i=kappa./Gc;

subplot(2,2,2);
plot(nu, kappa, 'b-','linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('\omega/2\pi (THz)','fontsize',font_size);
ylabel('\kappa(\omega) (W/m/K/THz)','fontsize',font_size);
ylim([0,200]);
xlim([0,52]);
set(gca,'ticklength',get(gca,'ticklength')*3,'xtick',0:10:50);
title('(b)');

subplot(2,2,3);
plot(nu(1:5000),lambda_i(1:5000),'b-','linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('\omega/2\pi (THz)','fontsize',font_size);
ylabel('\lambda(\omega) (nm)','fontsize',font_size);
ylim([0,10000]);
xlim([0,52]);
set(gca,'ticklength',get(gca,'ticklength')*3,'xtick',0:10:50);
title('(c)');

len=10.^(1:0.1:6);
for l=1:length(len)
    tmp=kappa./(1+lambda_i/len(l));
    k_L(l)=sum(tmp(1:5000))*(nu(2)-nu(1));
end

subplot(2,2,4);
semilogx(len/1000,k_L,'b-','linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('L (\mum)','fontsize',font_size);
ylabel('\kappa (W/mK)','fontsize',font_size);
ylim([0,3200]);
xlim([1.0e-2,10^3]);
set(gca,'ticklength',get(gca,'ticklength')*3,'xtick',10.^(-2:1:6));
title('(d)');

