clear; close all;
load force_gpu.out; load force_cpu.out;

% The difference should be of the order of 1.0e-5 (used float32 in GPU)
figure;
plot(force_gpu-force_cpu);
xlabel('force components');
ylabel('force difference (eV/A)');

