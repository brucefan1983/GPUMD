clear;close all;

load nep4/loss.out;loss4=loss;
load nep4/force_test.out;f4=force_test;
f4nep=reshape(f4(:,1:3),size(f4,1)*3,1);
f_dft=reshape(f4(:,4:6),size(f4,1)*3,1);

load nep4type/loss.out;loss4type=loss;
load nep4type/force_test.out;f4type=force_test;
f4type_nep=reshape(f4type(:,1:3),size(f4type,1)*3,1);

load nep3/loss.out;loss3=loss;
load nep3/force_test.out;f3=force_test;
f3nep=reshape(f3(:,1:3),size(f3,1)*3,1);

figure;
subplot(1,2,1);
loglog(loss4(:,6),'linewidth',2);hold on;
loglog(loss4type(:,6),'linewidth',2);
loglog(loss4(:,9),'linewidth',2);
loglog(loss4type(:,9),'linewidth',2);
xlabel('step/100');
ylabel('Force RMSE');
set(gca,'fontsize',15,'linewidth',1);
legend('NEP4(d_{ij}=1)-train','NEP4(d_{ij} trainable)-train','NEP4(d_{ij}=1)-test','NEP4(d_{ij} trainable)-test');
axis tight

subplot(1,2,2);
plot(f_dft,f4nep,'x','linewidth',2);hold on;
plot(f_dft,f4type_nep,'+','linewidth',2);hold on;
plot(linspace(-10,10,100),linspace(-10,10,100),'linewidth',2);
xlabel('DFT force (eV/A)');
ylabel('Test NEP force (eV/A)');
set(gca,'fontsize',15,'linewidth',1);
legend('NEP4(d_{ij}=1)','NEP4(d_{ij} trainable)');
axis tight



