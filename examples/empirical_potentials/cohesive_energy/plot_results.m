clear; close all; font_size = 15;
% some parameters of the model:
a=5.4334;
N=8000;
% load the data
load cohesive.out;
% plot the results
figure;
plot(cohesive(:,1)*a,cohesive(:,2)/N,'o-','linewidth',2);
xlabel('Lattice Constant (\AA)','fontsize',font_size,'interpreter','latex');
ylabel('Energy (eV/atom)','fontsize',font_size,'interpreter','latex');
set(gca,'fontsize',font_size);
axis tight;
