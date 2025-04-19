clear; close all;

% Call my_snes to evolve
dim = 5;
[best_fitness, elite] = my_snes(dim, 1000);
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


disp(elite(end,:))

load energy_train;
load counts;
energy_train=energy_train.*sum(counts,2);

for m=1:size(counts,2)
    energy_train(:,2)=energy_train(:,2)+counts(:,m)*elite(end,m);
end

energy_train=energy_train./sum(counts,2);


figure;
plot(energy_train(:,2),energy_train(:,1),'.','markersize',5); hold on;
plot(linspace(-10,0,100),linspace(-10,0,100))
xlabel('DFT energy (eV/atom)','fontsize',15,'interpreter','latex');
ylabel('NEP energy (eV/atom)','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);



