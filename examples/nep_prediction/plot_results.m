clear; %close all;
load energy_train.out; load force_train.out; 

figure;
plot(energy_train(:,2),energy_train(:,1),'.','markersize',20); hold on;
plot(-3.9:0.01:-3.65,-3.9:0.01:-3.65,'linewidth',2);
xlabel('DFT energy (eV/atom)','fontsize',15,'interpreter','latex');
ylabel('NEP energy (eV/atom)','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
axis tight;

figure;
plot(force_train(:,4:6),force_train(:,1:3),'.','markersize',20); hold on;
plot(-4:0.01:4,-4:0.01:4,'linewidth',2);
xlabel('DFT force (eV/$\AA$)','fontsize',15,'interpreter','latex');
ylabel('NEP force (eV/$\AA$)','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
axis tight;
