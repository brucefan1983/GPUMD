clear;close all;
load dipole_test.out; 

figure;
plot(dipole_test(:,1),dipole_test(:,2),'.','markersize',30);hold on;
plot(-0.4:0.01:0.4,-0.4:0.01:0.4);
xlabel('CCSD dipole/atom','interpreter','latex');
ylabel('NEP dipole/atom','interpreter','latex');
set(gca,'fontsize',15,'linewidth',1.5);
grid on;
axis equal
axis tight
