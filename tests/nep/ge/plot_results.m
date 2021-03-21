clear; %close all;
load energy.out;
load virial.out;
load force.out;

strain=[0.001:0.001:0.01, 0.012:0.002:0.02, 0.025:0.005:0.05];
strain=[-fliplr(strain), 0, strain];

figure;
subplot(1,3,1);
plot(force(:,4:6), force(:,1:3),'.'); hold on;
plot(-0.6:0.1:0.6,-0.6:0.1:0.6,'r-');
xlabel('Training force (eV/$\AA$)','fontsize',12,'interpreter','latex');
ylabel('NEP force (eV/$\AA$)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

subplot(1,3,2);
plot(strain,energy(16:58,2),'bo','linewidth',1);hold on;
plot(strain,energy(16:58,1),'b-','linewidth',1);hold on;
plot(strain,energy(59:101,2),'ro','linewidth',1);hold on;
plot(strain,energy(59:101,1),'r-','linewidth',1);hold on;
plot(strain,energy(102:144,2),'go','linewidth',1);hold on;
plot(strain,energy(102:144,1),'g-','linewidth',1);hold on;
xlabel('Strain','fontsize',12,'interpreter','latex');
ylabel('Energy (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;
legend('DFT','NEP');

subplot(1,3,3);
plot(strain,virial(16:58,2),'bo','linewidth',1);hold on;
plot(strain,virial(16:58,1),'B-','linewidth',1);hold on;
xlabel('Strain','fontsize',12,'interpreter','latex');
ylabel('Virial (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;
legend('DFT','NEP');

