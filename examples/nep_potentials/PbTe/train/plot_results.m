clear; close all;
load energy_test.out;load virial_test.out;load force_test.out; load loss.out; 

figure;
plot(energy_test(:,2),energy_test(:,1),'.','markersize',20); hold on;
plot(-3.9:0.01:-3.65,-3.9:0.01:-3.65,'linewidth',2);
xlabel('DFT energy (eV/atom)','fontsize',15,'interpreter','latex');
ylabel('NEP energy (eV/atom)','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
axis tight;

figure;
plot(force_test(:,4:6),force_test(:,1:3),'.','markersize',20); hold on;
plot(-4:0.01:4,-4:0.01:4,'linewidth',2);
xlabel('DFT force (eV/$\AA$)','fontsize',15,'interpreter','latex');
ylabel('NEP force (eV/$\AA$)','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
axis tight;

figure;
loglog(loss(:,1),loss(:,2:6),'-','linewidth',2); hold on;
loglog(loss(:,1),loss(:,8:9),'-','linewidth',4); hold on;
xlabel('Generation','fontsize',15,'interpreter','latex');
ylabel('Loss functions','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
legend('Total','L1-Reg','L2-Reg','Energy-train','Force-train','Energy-test','Force-test');
axis tight
