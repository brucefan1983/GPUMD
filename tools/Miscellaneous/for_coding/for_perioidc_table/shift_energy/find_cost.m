function [cost] = find_cost(population)
N_pop = size(population, 1);
cost = zeros(N_pop, 1);
load energy_train;
load counts;


for n_pop = 1 : N_pop
    energy=energy_train.*sum(counts,2);

    para=population(n_pop,:);
    for m=1:size(counts,2)
        energy(:,2)=energy(:,2)+counts(:,m)*para(m);
    end

    energy=energy./sum(counts,2);

    cost(n_pop) = mean((energy(:,1)-energy(:,2)).^2) + 0.0e-2 * mean(para.^2) + 0.00e-2 * mean(abs(para));
end
