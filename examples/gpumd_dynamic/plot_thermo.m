clear; close all;

load thermo.out;

t = (1:size(thermo,1))*0.01;

figure;
plot(t, thermo(:,1));
xlabel('Time (ps)');
ylabel('Temperature (K)');

