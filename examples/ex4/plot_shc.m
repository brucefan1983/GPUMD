clear; close all; font_size = 10;
load shc.out; shc=shc(end/2+1:end,:);

% input parameters for SHC
dt=2; %fs
Nc=250;
M=100000;

% calculated parameters
dt_in_ps = dt/1000;  % ps
nu=(0.1:0.1:60);     % THz
Ns=length(shc)/Nc;   % number of simulations

% reshape
ki=reshape(shc(:,1),Nc,Ns).';
ko=reshape(shc(:,2),Nc,Ns).';
% current
ji=ki(:,1);
jo=ko(:,1);

% mean time correlation function
ki=mean(ki,1)*1000/10.18; %eV/ps
ko=mean(ko,1)*1000/10.18; %eV/ps

% does not need 0.5 ps
Nc = 250;
ki=ki(1:Nc);
ko=ko(1:Nc);
time_in_ps = (0:Nc-1)*dt_in_ps;

figure
subplot(1,2,1);
plot(time_in_ps,ki,'b-', time_in_ps,ko,'r--','linewidth',2);
set(gca,'fontsize', font_size);
xlabel('Correlation time (ps)','fontsize', font_size);
ylabel('K (eV/ps)','fontsize', font_size);
xlim([0, 0.3]);
legend('in','out');
title('(a)');

% use K(-t) = K(t) symmetry
ki=ki.*[1,2*ones(1,Nc-1)];
ko=ko.*[1,2*ones(1,Nc-1)];

% Hann window
ki=ki.*(cos(pi*(0:Nc-1)/Nc)+1)*0.5;
ko=ko.*(cos(pi*(0:Nc-1)/Nc)+1)*0.5;

% correction for possible numerical errors
ki=ki-mean(ki);
ko=ko-mean(ko);

% the Fourier transform
qi=zeros(length(nu),1);
qo=zeros(length(nu),1);
% use discrete cosine transform
for n=1:length(nu)
   qi(n)=2*dt_in_ps*sum(ki.*cos(2*pi*nu(n)*(0:Nc-1)*dt_in_ps));
   qo(n)=2*dt_in_ps*sum(ko.*cos(2*pi*nu(n)*(0:Nc-1)*dt_in_ps));
end

DT=20; 
A=0.142*sqrt(3)*40*0.335; % nm^2
Gic=160*qi/A/DT;
Goc=160*qo/A/DT;

subplot(1,2,2);
plot(nu, Gic, 'b-', nu, Goc,'r--', 'linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('\omega/2\pi (THz)','fontsize',font_size);
ylabel('g(\omega) (GW/m^2/K/THz)','fontsize',font_size);
ylim([0,0.4]);
xlim([0,52]);
legend('in','out');
title('(b)');

% Results from temperature.out
G = 10.2079; % GW/m^2K

% this should be close to zero:
1 - ( sum(Gic+Goc) * (nu(2)-nu(1)) ) / G

