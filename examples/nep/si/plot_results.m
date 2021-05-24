clear; close all;
load energy.out;load virial.out;load force.out; load loss.out; load ann.out;

figure;

subplot(2,2,1);
plot(energy(:,2),energy(:,1),'.'); hold on;
plot(-4.8:0.01:-3.8,-4.8:0.01:-3.8);
xlabel('DFT energy (eV/atom)','fontsize',12,'interpreter','latex');
ylabel('NEP energy (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

subplot(2,2,2);
plot(force(:,4:6),force(:,1:3),'.'); hold on;
plot(-4:0.01:4,-4:0.01:4);
xlabel('DFT force (eV/$\AA$)','fontsize',12,'interpreter','latex');
ylabel('NEP force (eV/$\AA$)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

subplot(2,2,3);
plot(virial(:,2),virial(:,1),'.'); hold on;
plot(-3:0.01:6,-3:0.01:6);
xlabel('DFT virial (eV/atom)','fontsize',12,'interpreter','latex');
ylabel('NEP virial (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

subplot(2,2,4);
plot(ann);
xlabel('generation/100','fontsize',12,'interpreter','latex');
ylabel('ANN parameters','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight

figure;
semilogy(loss(5:5:end,1)/100,sum(loss(5:5:end,5),2),'d-','linewidth',1); hold on;
plot(loss(5:5:end,1)/100,sum(loss(5:5:end,6),2),'s-','linewidth',1);
plot(loss(5:5:end,1)/100,sum(loss(5:5:end,7),2),'o-','linewidth',1);
plot(loss(5:5:end,1)/100,sum(loss(5:5:end,3),2),'<-','linewidth',1); hold on;
plot(loss(5:5:end,1)/100,sum(loss(5:5:end,4),2),'>-','linewidth',1); hold on;
plot(loss(5:5:end,1)/100,sum(loss(5:5:end,2),2),'*-','linewidth',1); hold on;
xlabel('generation/100','fontsize',14,'interpreter','latex');
ylabel('Loss functions','fontsize',14,'interpreter','latex');
set(gca,'fontsize',14,'ticklength',get(gca,'ticklength')*2);
legend('Energy','Force','Virial','L1-Reg','L2-Reg','Total');
ylim([0.001,0.5]);

