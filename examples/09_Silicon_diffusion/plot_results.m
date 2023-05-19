clear; close all; font_size = 12;
load sdc.out; 
load msd.out; 

figure;
subplot(2,2,1);
plot(sdc(:,1),mean(sdc(:,2:4),2),'linewidth',3);
xlabel('Time (ps)');
ylabel('VAC (A^2/ps^2)');
set(gca,'fontsize',15);
title('(a)');
grid on;

subplot(2,2,2);
plot(msd(:,1),mean(msd(:,2:4),2),'linewidth',3);
xlabel('Time (ps)');
ylabel('MSD (A^2)');
set(gca,'fontsize',15);
title('(b)');
grid on;

subplot(2,2,3);
plot(sdc(:,1),mean(sdc(:,5:7),2),'-','linewidth',4);hold on;
plot(sdc(:,1),mean(msd(:,5:7),2),':','linewidth',3);hold on;
xlabel('Time (ps)');
ylabel('SDC (A^2/ps)');
legend('from VAC','from MSD');
set(gca,'fontsize',15);
title('(c)');
grid on;


