clear; close all;

% Call my_snes to evolve
dim = 130;
[best_fitness, elite] = my_snes(dim, 300);
num_generations = length(best_fitness);

% Evolution of the best fitness:
figure
loglog(1 : num_generations, best_fitness, 'linewidth',2)
xlabel('Generation','fontsize',12);
ylabel('Best Fitness','fontsize',12);
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);

% Evolution of the best solution:
figure
semilogx(1 : num_generations, elite)
xlabel('Generation','fontsize',12);
ylabel('Best Solution','fontsize',12);
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);


%-0.3389    8.4898    6.4209   -6.4508   -3.6579 0.5389
disp(elite(end,:))
load f_cal0;
load f_ref;
% compare with the training set:
[y, f_cal] = find_cost(elite(end,:));
figure;
plot(reshape(f_ref,64*3,1), reshape(f_cal,64*3,1), 'ro'); hold on;
plot(reshape(f_ref,64*3,1),reshape(f_cal0,64*3,1),'bx');
plot(linspace(-20,20,100),linspace(-20,20,100))
legend('new','old')
sqrt(mean(mean((f_cal-f_ref).^2)))
sqrt(mean(mean((f_cal0-f_ref).^2)))



