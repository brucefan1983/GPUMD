clear; close all;
load energy.out;
load virial.out;
load force.out; 

figure;
subplot(1,3,1);
plot(force(:,4:6),force(:,1:3),'.'); hold on;
plot(-8:0.1:8,-8:0.1:8);
xlabel('DFT force (eV/$\AA$)','fontsize',12,'interpreter','latex');
ylabel('NEP force (eV/$\AA$)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

subplot(1,3,2);
plot(energy(:,2),energy(:,1),'.'); hold on;
plot(-5:0.1:-2,-5:0.1:-2);
xlabel('DFT energy (eV/atom)','fontsize',12,'interpreter','latex');
ylabel('NEP energy (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

subplot(1,3,3);
plot(virial(:,2),virial(:,1),'.'); hold on;
plot(-4:0.1:7,-4:0.1:7);
xlabel('DFT virial (eV/atom)','fontsize',12,'interpreter','latex');
ylabel('NEP virial (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;


