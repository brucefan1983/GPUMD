clear; close all; font_size = 12;
load viscosity.out;

figure;
subplot(2,2,1);
plot(viscosity(:,1),viscosity(:,2:4),'linewidth',1); hold on;
plot(viscosity(:,1),mean(viscosity(:,2:4),2),'linewidth',3);
xlabel('Correlation time (ps)');
ylabel('Longitudinal SAC (eV^2)');
xlim([0,1]);
set(gca,'fontsize',15);
title('(a)');
grid on;

subplot(2,2,2);
plot(viscosity(:,1),1000*viscosity(:,11:13),'-','linewidth',3); hold on;
plot(viscosity(:,1),1000*mean(viscosity(:,11:13),2),'k','linewidth',4);
legend('xx','yy','zz','average');
xlim([0,1]);
xlabel('Correlation time (ps)');
ylabel('Longitudinal viscosity (mPa s)');
set(gca,'fontsize',15);
title('(b)');
grid on;


subplot(2,2,3);
plot(viscosity(:,1),viscosity(:,5:10),'linewidth',1); hold on;
plot(viscosity(:,1),mean(viscosity(:,5:10),2),'linewidth',3);
xlabel('Correlation time (ps)');
ylabel('Shear SAC (eV^2)');
xlim([0,1]);
set(gca,'fontsize',15);
title('(c)');
grid on;

subplot(2,2,4);
plot(viscosity(:,1),1000*viscosity(:,14:16),'-','linewidth',3); hold on;
plot(viscosity(:,1),1000*viscosity(:,17:19),':','linewidth',3); hold on;
plot(viscosity(:,1),1000*mean(viscosity(:,14:19),2),'k','linewidth',4);
legend('xy','xz','yz','yx','zx','zy','average');
xlim([0,1]);
xlabel('Correlation time (ps)');
ylabel('Shear viscosity (mPa s)');
set(gca,'fontsize',15);
title('(d)');
grid on;



