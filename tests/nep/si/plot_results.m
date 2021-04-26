clear; 
load energy.out;
load virial.out;
load force.out; 
load loss.out;
load ann.out

figure;

subplot(2,3,1);
plot(energy(:,2),energy(:,1),'.'); hold on;
plot(-4.8:0.01:-4.2,-4.8:0.01:-4.2);
xlabel('DFT energy (eV/atom)','fontsize',12,'interpreter','latex');
ylabel('NEP energy (eV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
axis tight;

subplot(2,3,2);
plot(force(:,4:6),force(:,1:3),'.'); hold on;
plot(-4:0.01:4,-4:0.01:4);
xlabel('DFT force (eV/$\AA$)','fontsize',12,'interpreter','latex');
ylabel('NEP force (eV/$\AA$)','fontsize',12,'interpreter','latex');
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
loglog(loss(:,1),loss(:,5),'rs-'); hold on;
loglog(loss(:,1),loss(:,8),'bo-'); hold on;
axis tight
xlabel('training step','fontsize',12,'interpreter','latex');
ylabel('Energy RMSE (meV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
legend('Train','Test');

subplot(2,3,5);
loglog(loss(:,1),loss(:,6),'rs-'); hold on;
loglog(loss(:,1),loss(:,9),'bo-');
axis tight
xlabel('training step','fontsize',12,'interpreter','latex');
ylabel('Force RMSE (meV/A)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
legend('Train','Test');

subplot(2,3,6);
loglog(loss(:,1),loss(:,7),'rs-'); hold on;
loglog(loss(:,1),loss(:,10),'bo-');
axis tight
xlabel('training step','fontsize',12,'interpreter','latex');
ylabel('Virial RMSE (meV/atom)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
legend('Train','Test');

figure;
semilogx(loss(:,1),ann);
xlabel('training step','fontsize',12,'interpreter','latex');
ylabel('ANN parameters','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
