clear; %close all;
load energy.out;
load virial.out;
load force.out;
N=256;

figure;
subplot(1,3,1);
plot(force(:,4), force(:,1),'bx');
xlabel('Training force (eV/$\AA$)','fontsize',12,'interpreter','latex');
ylabel('NN2B force (eV/$\AA$)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);

strain = -0.02:0.001:0.02;
subplot(1,3,2);
plot(strain, energy(:,2)/N,'ro','linewidth',1);hold on;
plot(strain, energy(:,1)/N,'bx','linewidth',1);hold on;
xlabel('Strain','fontsize',12,'interpreter','latex');
ylabel('Energy (eV/atom)','fontsize',12,'interpreter','latex');
legend('Training','NN2B');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);

subplot(1,3,3);
plot(strain, virial(1:end/6,2)/N,'ro','linewidth',1);hold on;
plot(strain, virial(1:end/6,1)/N,'bx','linewidth',1);hold on;
xlabel('Strain','fontsize',12,'interpreter','latex');
ylabel('Virial (eV/atom)','fontsize',12,'interpreter','latex');
legend('Training','NN2B');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);



