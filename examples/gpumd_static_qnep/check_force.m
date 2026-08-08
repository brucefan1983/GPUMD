clear; close all;
% dump.xyz is in extended XYZ format, so skip the two header lines and take the force columns
fid = fopen('dump.xyz'); fgetl(fid); fgetl(fid);
columns = textscan(fid, '%s %f %f %f %f %f %f', 250);
fclose(fid);
f_gpumd = [columns{5} columns{6} columns{7}];
load ../qnep_train/force_train.out; f_nep=force_train(end-250+1:end,1:3);

% The difference should be of the order of 1.0e-5 (used float32)
figure;
plot(f_gpumd-f_nep);
xlabel('force components');
ylabel('force difference (eV/A)');

