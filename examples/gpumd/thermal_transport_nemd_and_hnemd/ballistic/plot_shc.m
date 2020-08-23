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

figure
subplot(1,2,1);
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

DT=19; % smaller than the set value 20 K
A=0.142*sqrt(3)*100*0.335; % nm^2
Gc=160*q/A/DT;

subplot(1,2,2);
plot(nu,Gc,'b-','linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('\omega/2\pi (THz)','fontsize',font_size);
ylabel('g(\omega) (GW/m^2/K/THz)','fontsize',font_size);
ylim([0,0.4]);
xlim([0,52]);
title('(b)');

save('Gc','Gc');

G=sum(Gc)*(nu(2)-nu(1)) 

