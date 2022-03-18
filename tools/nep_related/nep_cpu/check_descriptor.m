clear; close all;
load descriptor.out;

figure;
plot(descriptor(:,:).','o-');hold on;
xlabel('descriptor components');
ylabel('descriptor value');
set(gca,'fontsize',15);



