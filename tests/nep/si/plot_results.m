clear; close all;
load energy.out;
load virial.out;
load force.out; 
load train.out;
load potential.out

figure;
subplot(2,3,1);
plot(force(:,4:6),force(:,1:3),'.'); hold on;
plot(-3:0.01:3,-3:0.01:3);
xlabel('DFT force (eV/$\AA$)','fontsize',12,'interpreter','latex');
ylabel('NEP force (eV/$\AA$)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

subplot(2,3,2);
plot(energy(:,2),energy(:,1),'.'); hold on;
plot(-4.8:0.01:-4.2,-4.8:0.01:-4.2);
xlabel('DFT energy (eV/atom)','fontsize',12,'interpreter','latex');
ylabel('NEP energy (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

subplot(2,3,3);
plot(virial(:,2),virial(:,1),'.'); hold on;
plot(-3:0.01:4,-3:0.01:4);
xlabel('DFT virial (eV/atom)','fontsize',12,'interpreter','latex');
ylabel('NEP virial (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;


subplot(2,3,4);
loglog(train(:,1),train(:,2),'-'); hold on;
xlabel('training step','fontsize',12,'interpreter','latex');
ylabel('Energy RMSE (meV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);

subplot(2,3,5);
loglog(train(:,1),train(:,3),'-'); hold on;
xlabel('training step','fontsize',12,'interpreter','latex');
ylabel('Force RMSE (meV/A)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);

subplot(2,3,6);
loglog(train(:,1),train(:,4),'-'); hold on;
xlabel('training step','fontsize',12,'interpreter','latex');
ylabel('Virial RMSE (meV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);

figure;
plot(train(:,1),potential);
xlabel('training step','fontsize',12,'interpreter','latex');
ylabel('ANN parameters','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);


