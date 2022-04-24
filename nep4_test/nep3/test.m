clear;close all;

load loss.out;
load force_test.out;

figure;
loglog(loss(:,4:end),'linewidth',2);
xlabel('step/100');
ylabel('loss');
set(gca,'fontsize',15,'linewidth',1);

figure;
plot(force_test(:,4:6),force_test(:,1:3),'.','linewidth',5);hold on;
plot(linspace(-10,10,100),linspace(-10,10,100),'linewidth',2);
xlabel('DFT force (eV/A)');
ylabel('NEP4 force (eV/A)');
set(gca,'fontsize',15,'linewidth',1);
axis tight
