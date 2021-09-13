clear; close all;
load energy.out;load virial.out;load force.out; load loss.out; 

figure;
plot(energy(:,2),energy(:,1),'.','markersize',20); hold on;
plot(-3.9:0.01:-3.65,-3.9:0.01:-3.65,'linewidth',2);
xlabel('DFT energy (eV/atom)','fontsize',15,'interpreter','latex');
ylabel('NEP energy (eV/atom)','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
axis tight;

figure;
plot(force(:,4:6),force(:,1:3),'.','markersize',20); hold on;
plot(-4:0.01:4,-4:0.01:4,'linewidth',2);
xlabel('DFT force (eV/$\AA$)','fontsize',15,'interpreter','latex');
ylabel('NEP force (eV/$\AA$)','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
axis tight;

figure;
loglog(loss(:,1),loss(:,2:6),'-','linewidth',2); hold on;
xlabel('Generation','fontsize',15,'interpreter','latex');
ylabel('Loss functions','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
legend('Total','L1-Reg','L2-Reg','Energy','Force');
axis tight
