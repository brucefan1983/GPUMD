clear; close all;
r0 = [0.0, 0.0, 0.5, 0.5, 0.25, 0.25, 0.75, 0.75; ...
      0.0, 0.5, 0.0, 0.5, 0.25, 0.75, 0.25, 0.75; ...
      0.0, 0.5, 0.5, 0.0, 0.25, 0.75, 0.75, 0.25].';
n0 = size(r0, 1);
nxyz = 8 * [1, 1, 1];
N = nxyz(1) * nxyz(2) * nxyz(3) * n0;
a = 3.57 * [1, 1, 1];
box_length = a .* nxyz;

r = zeros(N, 3);
n = 0;
for nx = 0 : nxyz(1) - 1
    for ny = 0 : nxyz(2) - 1
        for nz = 0 : nxyz(3) - 1
            for m = 1 : n0
                n = n + 1;
                r(n, :) = a .* ([nx,ny,nz] + r0(m, :));   
            end
        end
    end
end

fid = fopen('xyz.in', 'w');
fprintf(fid, '%d %d %g %d %d %d\n', N, 4, 2.1, 0, 0, 0);
fprintf(fid, '%d %d %d %g %g %g\n', 1, 1, 1, box_length);
for n =1 : N
    fprintf(fid, '%d %g %g %g %g\n', 0, r(n, :), 12);
end
fclose(fid);
