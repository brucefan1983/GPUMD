load thermo.out;
load thermo1.out;
close all;
figure;
plot((thermo(:,1)-thermo1(:,1)),'linewidth',4);hold on;