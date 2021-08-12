function [loss, y] = ann(x0,y0,population)
N_samples = length(x0);
N_neurons = 10;
N_pop = size(population, 1);
loss = zeros(N_pop, 1);
y = zeros(N_pop, N_samples);
for n_pop = 1 : N_pop
    para = population(n_pop, :);
    u = para(1:N_neurons)';
    offset = N_neurons;
    v = reshape(para(offset+1 : offset + N_neurons*N_neurons), N_neurons, N_neurons);
    offset = offset + N_neurons*N_neurons;
    w = para(offset+1 : offset + N_neurons);
    offset = offset + N_neurons;
    a = para(offset+1 : offset + N_neurons)';
    offset = offset + N_neurons;
    b = para(offset+1 : offset + N_neurons)';
    c = para(end);
    for n_sample = 1 : N_samples % loop over the samples
        y(n_pop, n_sample) = w * tanh(v * tanh(u * x0(n_sample) - a) - b) - c;
    end
    loss(n_pop) = 0.5 * mean((y(n_pop, :) - y0).^2) ...
        + 1.0e-5 * 0.5 * sum(para.^2) + 1.0e-5 * sum(abs(para));
end
