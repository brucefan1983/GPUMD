clear;  close all; font_size=10; 
load temperature.out; 

% Some parameters from MD
dt = 0.001; % ps
Ns = 1000;  % sample interval
N_temp = size(temperature, 1);
temp_difference = 20;

% temperature profile
figure;

subplot(1, 2, 1);
plot(mean(temperature(end/2+1:end,2:end-2)),'bo-','linewidth',2);
xlabel('group index','fontsize',font_size);
ylabel('T (K)','fontsize',font_size);
set(gca,'fontsize',font_size);
title('(a)');

% energy exchange between the system and the thermostats
subplot(1, 2, 2);
t=dt*(1:N_temp)* Ns/1000; % ns
plot(t,temperature(:,end-1)/1000,'r-','linewidth',2);
hold on;
plot(t,temperature(:,end)/1000,'b--','linewidth',2);
hold on;
xlabel('t (ns)','fontsize',font_size);
ylabel('Heat (keV)','fontsize',font_size);
set(gca,'fontsize',font_size);
legend('source','sink');
title('(b)');

% heat flux calculated from the thermostats
Q1=(temperature(end/2,end-1)-temperature(end,end-1))/(N_temp/2)/dt/Ns;
Q2=(temperature(end,end)-temperature(end/2,end))/(N_temp/2)/dt/Ns;
Q=(Q1+Q2)/2 % eV/ps

% classical ballistic conductance
A=0.335*40*0.142*sqrt(3); % nm^2
G=160*Q/A/temp_difference % GW/m^2K

