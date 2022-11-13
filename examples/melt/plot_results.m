clear; close all; font_size = 12;
load thermo.out; 

time=(1:size(thermo,1))/5;
mean(thermo(end/2+1:end,1))

figure;
subplot(1,2,1);
plot(time,thermo(:,1));
xlabel('Time (ps)');
ylabel('Temperature (K)');
set(gca,'fontsize',15);

subplot(1,2,2);
plot(time,thermo(:,4:6))
xlabel('Time (ps)');
ylabel('Pressure (GPa)');
set(gca,'fontsize',15);



