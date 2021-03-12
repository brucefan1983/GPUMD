clear; close all;

[r0, box0] = find_r0_box0();
N = size(r0, 1);
strain = -0.02:0.001:0.02;
N_strain = length(strain);
N_force = 10;

fid = fopen('train.in', 'w');
fprintf(fid,'%d %d\n',N_force + N_strain, N_force);
for n=1:(N_force + N_strain)
    fprintf(fid,'%d\n',N);
end

for n = 1 : N_force
    r = r0 + rand(N, 3) * 0.05 * n;
    box = box0;
    [junk_energy, junk_virial, force] = find_E(r, box);
    fprintf(fid,'%f %f %f %f %f %f %f %f %f\n',box);
    for i = 1 : N
        fprintf(fid,'0 %f %f %f %f %f %f\n',r(i,:), force(i,:));
    end
end

figure;
plot(force)

energy = zeros(1, N_strain);
virial = zeros(6, N_strain);
for n = 1 : N_strain
    r = r0 * (1 + strain(n));
    box = box0 * (1 + strain(n));
    [energy(n), virial(:, n), junk_force] = find_E(r, box);
    fprintf(fid,'%f %f %f %f %f %f %f\n',energy(n), virial(:,n));
    fprintf(fid,'%f %f %f %f %f %f %f %f %f\n',box);
    for i = 1 : N
        fprintf(fid,'0 %f %f %f\n',r(i,:));
    end
end
fclose(fid);

figure;
plot(strain, energy/N);

figure;
plot(strain, virial(1, :)/N);
