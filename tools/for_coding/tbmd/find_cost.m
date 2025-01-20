function [cost, f_cal] = find_cost(population)
N_pop = size(population, 1);
cost = zeros(N_pop, 1);
load r;
load f_ref;
N_atom=size(r,1);
L=[9.483921 9.483921 9.483921];
for n_pop = 1 : N_pop
    para=population(n_pop,:);
    [energy,f_cal]=find_force_train(N_atom,L,r,para);
    cost(n_pop) = sqrt(mean(mean((f_cal - f_ref).^2))) ...
        + 1.0e-2 * mean(para.^2) + 1.0e-2 * mean(abs(para));
end
