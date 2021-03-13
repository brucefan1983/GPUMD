clear; %close all;
load energy.out;
load virial.out;
load force.out;
N=256;

figure;
subplot(1,2,1);
plot(force(:,4), force(:,1),'b.');
xlabel('Training force (eV/$\AA$)','fontsize',12,'interpreter','latex');
ylabel('NN2B force (eV/$\AA$)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);

subplot(1,2,2);
plot(energy(:,2)/N,energy(:,1)/N,'ro','linewidth',1);hold on;
xlabel('Strain','fontsize',12,'interpreter','latex');
ylabel('Energy (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);




