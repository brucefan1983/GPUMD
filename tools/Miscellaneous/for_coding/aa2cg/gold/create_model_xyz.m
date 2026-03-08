clear; close all;
r0 = [0.0, 0.0, 0.5, 0.5; ...
      0.0, 0.5, 0.0, 0.5; ...
      0.0, 0.5, 0.5, 0.0].';
n0 = size(r0, 1);
nxyz = 8 * [1, 1, 1];
N = nxyz(1) * nxyz(2) * nxyz(3) * n0;
a = 4.16 * [1, 1, 1];
box_length = a .* nxyz;

r = zeros(N, 3);
t = zeros(N, 1);
n = 0;
b = 0;
for nx = 0 : nxyz(1) - 1
    for ny = 0 : nxyz(2) - 1
        for nz = 0 : nxyz(3) - 1
            for m = 1 : n0
                n = n + 1;
                r(n, :) = a .* ([nx,ny,nz] + r0(m, :));   
                t(n) = b;
            end
            b = b + 1;
        end
    end
end

fid=fopen('model.xyz','w');
fprintf(fid,'%d\n',N);
fprintf(fid,'pbc=\"T T T\" Lattice=\"%g 0 0 0 %g 0 0 0 %g\" Properties=species:S:1:pos:R:3:group:I:1\n',box_length);

for n =1 : N
    fprintf(fid, 'Au %g %g %g %d\n', r(n, :), t(n));
end
fclose(fid);
