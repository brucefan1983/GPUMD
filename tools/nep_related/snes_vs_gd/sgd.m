clear; %close all;
x0 = 1 : 0.01 : 3;
N_samples = length(x0);
U0 = 10./x0.^12 - 10./x0.^6 + rand(1, N_samples) * 0.1;
U = zeros(1, N_samples);
N_neurons = 10; % 1-10-10-1 NN
N_neurons2 = N_neurons * N_neurons;
N_para = N_neurons * (N_neurons + 4) + 1;
para = (rand(N_para, 1) - 0.5) * 5;
para_tmp = para;
momentum = zeros(N_para, 1);
beta = 0.9;
learning_rate = 0.01;
N_steps = 10000;
rmse_U = zeros(N_steps, 1);
for step = 1 : N_steps
    para_d = zeros(N_para, 1);
    para_tmp = para + beta * momentum;
    for n_sample = 1 : N_samples
        % get the parameters:
        u = para_tmp(1:N_neurons); offset = N_neurons;
        v = reshape(para_tmp(offset+1 : offset + N_neurons*N_neurons), N_neurons, N_neurons); offset = offset + N_neurons*N_neurons;
        w = para_tmp(offset+1 : offset + N_neurons).'; offset = offset + N_neurons;
        a = para_tmp(offset+1 : offset + N_neurons); offset = offset + N_neurons;
        b = para_tmp(offset+1 : offset + N_neurons);
        c = para_tmp(end);
        % propagate the NN:
        y = tanh(u * x0(n_sample) - a);
        yd = 1 - y .* y;
        z = tanh(v * y - b);
        zd = 1 - z .* z;
        U(n_sample) = w * z - c;
        % calculate the derivatives
        deltaU = U(n_sample) - U0(n_sample);
        para_d1 = zeros(N_para, 1);
        para_d1(1:N_neurons) = deltaU * x0(n_sample) * yd .* (v.' * zd .* w.'); offset = N_neurons;
        para_d1(offset+1 : offset + N_neurons*N_neurons) = reshape(deltaU * (w.' .* zd) * y.', N_neurons*N_neurons, 1); offset = offset + N_neurons*N_neurons;
        para_d1(offset+1 : offset + N_neurons) = deltaU * z.'; offset = offset + N_neurons;
        para_d1(offset+1 : offset + N_neurons) = - deltaU * yd .* (v.' * zd .* w.'); offset = offset + N_neurons;
        para_d1(offset+1 : offset + N_neurons) = - deltaU * w.' .* zd;
        para_d1(end) = - deltaU;
        para_d = para_d + para_d1;
    end
    momentum = beta * momentum - (learning_rate / N_samples) * para_d;
    para = para + momentum;
    rmse_U(step) = sqrt(mean((U - U0).^2));
    if mod(step, 1000) == 0
        disp(['Step = ', num2str(step), ', best fitness = ', num2str(rmse_U(step))]);
    end
end

figure;semilogy(1:N_steps, rmse_U, '-');
figure;plot(para);
figure;
plot(x0, U0, 'ro', x0, U, 'bx');
legend('training data');
xlabel('x', 'fontsize', 15);
ylabel('y', 'fontsize', 15);
set(gca, 'fontsize', 15);
legend('raw', 'fit');
