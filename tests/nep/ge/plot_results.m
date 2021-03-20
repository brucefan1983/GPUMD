clear; %close all;
load energy.out;
load virial.out;
load force.out;

figure;
subplot(1,3,1);
plot(force(:,4:6), force(:,1:3),'o'); hold on;
plot(-0.6:0.1:0.6,-0.6:0.1:0.6,'r-');
xlabel('Training force (eV/$\AA$)','fontsize',12,'interpreter','latex');
ylabel('NN2B force (eV/$\AA$)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

subplot(1,3,2);
plot(energy(16:58,2),'bo','linewidth',1);hold on;
plot(energy(16:58,1),'bx','linewidth',1);hold on;
plot(energy(59:101,2),'ro','linewidth',1);hold on;
plot(energy(59:101,1),'rx','linewidth',1);hold on;
plot(energy(102:144,2),'go','linewidth',1);hold on;
plot(energy(102:144,1),'gx','linewidth',1);hold on;
%plot(-1.3:0.1:-1,-1.3:0.1:-1,'r-');
xlabel('Training energy (eV/atom)','fontsize',12,'interpreter','latex');
ylabel('NN2B energy (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

subplot(1,3,3);
plot(virial(:,2),virial(:,1),'b.','linewidth',1);hold on;
%plot(-1:0.1:1,-1:0.1:1,'r-');
xlabel('Training virial (eV/atom)','fontsize',12,'interpreter','latex');
ylabel('NN2B virial (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

