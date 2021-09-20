clear; close all;
r0 = [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0; ...
      0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0; ...
      0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5].';
type0 = [1 1 1 1 0 0 0 0];
mass0 = [207 207 207 207 128 128 128 128];
n0 = size(r0, 1);
nxyz = 10 * [1, 1, 1];
N = nxyz(1) * nxyz(2) * nxyz(3) * n0;
a = 6.5704 * [1, 1, 1];
box_length = a .* nxyz;

r = zeros(N, 3);
type = zeros(N, 1);
mass = zeros(N, 1);
n = 0;
for nx = 0 : nxyz(1) - 1
    for ny = 0 : nxyz(2) - 1
        for nz = 0 : nxyz(3) - 1
            for m = 1 : n0
                n = n + 1;
                r(n, :) = a .* ([nx,ny,nz] + r0(m, :));   
                type(n) = type0(m);
                mass(n) = mass0(m);
            end
        end
    end
end

fid = fopen('xyz.in', 'w');
fprintf(fid, '%d %d %g 0 0 0\n', N, 500, 9.0);
fprintf(fid, '%d %d %d %g %g %g\n', 1, 1, 1, box_length);

figure;
hold on;
for n =1 : N
    fprintf(fid, '%d %g %g %g %g\n', type(n), r(n, :), mass(n));
    if type(n)==52
        plot3(r(n, 1),r(n, 2),r(n, 3),'ro');
    else
        plot3(r(n, 1),r(n, 2),r(n, 3),'bo');
    end
end
fclose(fid);




