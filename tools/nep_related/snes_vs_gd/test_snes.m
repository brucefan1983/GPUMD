clear; %close all;
[x0,y0,N_neu,N_par]=get_inputs();
N_gen=1000;
[loss] = snes(x0,y0,N_par,N_gen);
figure
loglog(1:N_gen,loss,'linewidth',2)
xlabel('Generation','fontsize',12);
ylabel('Best Fitness','fontsize',12);
set(gca,'fontsize',12);

