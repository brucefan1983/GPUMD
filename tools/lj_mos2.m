% Pre-compute some LJ parameters of the REBO-LJ potentail for MoS2 
% These numbers are used in the src/rebo_mos2.cu file
clear; close all;
EPSILON=[0.00058595,0.02,sqrt(0.00058595*0.02)]; % epsilon (MM, SS, MS)
SIGMA=[4.20,3.13,3.665]; % sigma (MM, SS, MS)
R1=[3.5,2.3,2.75]; % inner cutoff of REBO (MM, SS, MS)
format long;
LJCUT1=SIGMA*0.95
LJCUT2=SIGMA*2.5
S12E4=4*EPSILON.*SIGMA.^12
S6E4=4*EPSILON.*SIGMA.^6
S12E48=48*EPSILON.*SIGMA.^12
S6E24=24*EPSILON.*SIGMA.^6
R=LJCUT1-R1;
U95=S12E4./LJCUT1.^12-S6E4./LJCUT1.^6;
F95=S6E24./LJCUT1.^7-S12E48./LJCUT1.^13;
D2=(3*U95./R-F95)./R
D3=(F95./R-2*U95./R./R)./R

% check the potential
figure;
for n=1:3
    r1=R1(n):0.001:LJCUT1(n);
    U1=D2(n)*(r1-R1(n)).^2+D3(n)*(r1-R1(n)).^3;
    r2=LJCUT1(n):0.001:LJCUT2(n);
    U2=S12E4(n)./r2.^12-S6E4(n)./r2.^6;
    plot(r1,U1,'-','linewidth',2);
    hold on;
    plot(r2,U2,'--','linewidth',2);
end
xlabel('r (A)');
ylabel('U (eV)');
legend('Mo-Mo','S-S','Mo-S');
set(gca,'fontsize',12);
