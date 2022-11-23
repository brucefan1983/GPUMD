clear; close all; font_size = 12;
load viscosity.out; 

figure;
subplot(1,2,1);
plot(viscosity(:,1),viscosity(:,2:7),'linewidth',1); hold on;
plot(viscosity(:,1),mean(viscosity(:,2:7),2),'linewidth',3);
xlabel('Time (ps)');
ylabel('SAC (eV^2)');
xlim([0,1]);
set(gca,'fontsize',15);
title('(a)');
grid on;

subplot(1,2,2);
plot(viscosity(:,1),1000*viscosity(:,8:10),'-','linewidth',3); hold on;
plot(viscosity(:,1),1000*viscosity(:,11:13),':','linewidth',3); hold on;
plot(viscosity(:,1),1000*mean(viscosity(:,8:13),2),'k','linewidth',4);
legend('xy','xz','yz','yx','zx','zy','average');
xlim([0,1]);
xlabel('Time (ps)');
ylabel('Viscosity (10^{-3}Pa s)');
set(gca,'fontsize',15);
title('(b)');
grid on;



