clear; close all; font_size = 10;
load shc.out; shc=sum(shc(end/10+1:end,:),2);

% input parameters for SHC
dt=2; %fs
Nc=250;
M=100000;

% calculated parameters
dt_in_ps = dt/1000;  % ps
nu=(0.01:0.01:60);   % THz
Ns=length(shc)/Nc;   % number of simulations

% reshape
k=reshape(shc(:,1),Nc,Ns).';

% mean time correlation function
k=mean(k,1)*1000/10.18; %eV/ps
Nc=250;
k=k(1:Nc);
time_in_ps=(0:Nc-1)*dt_in_ps;

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
A=0.142*sqrt(3)*200*0.335; % nm^2
k=16*q/A/T/Fe;

load ../ballistic/Gc;
lambda_i=k./Gc;

subplot(2,2,2);
plot(nu, k, 'b-','linewidth',1.5);
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
ylim([0,6000]);
xlim([0,52]);
set(gca,'ticklength',get(gca,'ticklength')*3,'xtick',0:10:50);
title('(c)');

len=10.^(1:0.1:6);
for l=1:length(len)
    tmp=k./(1+lambda_i/len(l));
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

