clear; close all;
load force.out; f_gpumd=force;
load ../train/force_test.out; f_nep=force_test(end-250+1:end,1:3);

% The difference should be of the order of 1.0e-5 (used float32)
figure;
plot(f_gpumd-f_nep);
xlabel('force components');
ylabel('force difference (eV/A)');

