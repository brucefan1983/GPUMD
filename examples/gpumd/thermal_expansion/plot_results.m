clear; close all; font_size = 12;
load thermo.out;

time = 0.01*(1:length(thermo)); % ps
temp=100:100:1000; % temperature points (input)
NT=length(temp); % number of temperature points
NC=10; % number of cells in each direction
M=length(thermo)/NT;

% temperature vs time
figure;

subplot(2,2,1);
plot(time, thermo(:,1), 'linewidth',2);
xlabel('Time (ps)','fontsize',font_size);
ylabel('Temperature (K)','fontsize',font_size);
xlim([0,200]);
ylim([0,1100]);
set(gca,'fontsize',font_size,'linewidth',1,'ticklength',get(gca,'ticklength')*2);
title('(a)');

% pressure vs time
subplot(2,2,2);
plot(time, mean(thermo(:,4:6),2), 'linewidth',2);
xlabel('Time (ps)','fontsize',font_size);
ylabel('Pressure (GPa)','fontsize',font_size);
xlim([0,200]);
ylim([-0.1,0.4]);
set(gca,'fontsize',font_size,'linewidth',1,'ticklength',get(gca,'ticklength')*2);
title('(b)')


a=mean(thermo(:,7:9),2)/NC;
% lattice constant vs time
subplot(2,2,3);
plot(time, a, 'linewidth',2);
xlabel('Time (ps)','fontsize',font_size);
ylabel('a (Angstrom)','fontsize',font_size);
xlim([0,200]);
ylim([5.43,5.48]);
set(gca,'fontsize',font_size,'linewidth',1,'ticklength',get(gca,'ticklength')*2);
title('(c)')

a=reshape(a,M,NT);
a=mean(a(end/2+1:end,:),1);
p=polyfit(temp,a,1);
expansion_coefficient=p(1)/p(2)

% thermal expansion
subplot(2,2,4);
plot(temp,a,'o','linewidth',2,'markersize',8);
hold on;
plot(p(1)*(1:1100)+p(2),'r-');
xlabel('Temperature (K)','fontsize',font_size);
ylabel('a (Angstrom)','fontsize',font_size);
xlim([0,1100]);
ylim([5.43,5.48]);
set(gca,'fontsize',font_size,'linewidth',1,'ticklength',get(gca,'ticklength')*2);
title('(d)');


