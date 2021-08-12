function [best_fitness, elite] = my_snes(number_of_variables, maximal_generation)
population_size = 4 + floor(3 * log(number_of_variables));
best_fitness = ones(maximal_generation, 1);
elite = zeros(maximal_generation, number_of_variables);
mu = 2*rand(1, number_of_variables)-1; % initial mean
sigma = ones(1, number_of_variables); % initial variance
learn_rates = [1, (3 + log(number_of_variables))/(5 * sqrt(number_of_variables)) / 2];
utility = max(0, log(population_size/2+1)-log(1:population_size));
utility = utility / sum(utility) - 1/population_size; % sum of utility is zero
for generation = 1 : maximal_generation
    s = randn(population_size, number_of_variables);
    population = repmat(mu, population_size, 1) + repmat(sigma, population_size, 1) .* s;
    cost = ann(population, 1, 0);
    [cost, index] = sort(cost);
    s = s(index, :);
    population = population(index, :);
    best_fitness(generation) = cost(1);
    elite(generation, :) = population(1, :);
    mu = mu + learn_rates(1) * sigma .* (utility * s); % update mean
    sigma = sigma .* exp(learn_rates(2) * (utility * (s .* s - 1))); % update variance
    if mod(generation, 100) == 0
        disp(['Generation = ', num2str(generation), ', best fitness = ', num2str(cost(1))]);
    end
end
