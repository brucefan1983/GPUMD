clear; close all;
r0 = [0.0, 0.0, 0.5, 0.5, 0.25, 0.25, 0.75, 0.75; ...
      0.0, 0.5, 0.0, 0.5, 0.25, 0.75, 0.25, 0.75; ...
      0.0, 0.5, 0.5, 0.0, 0.25, 0.75, 0.75, 0.25].';
n0 = size(r0, 1);
nxyz = 10 * [1, 1, 1];
N = nxyz(1) * nxyz(2) * nxyz(3) * n0;
a = 5.43 * [1, 1, 1];
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

fid=fopen('model.xyz','w');
fprintf(fid,'%d\n',N);
fprintf(fid,'triclinic=F pbc=\"T T T\" Lattice=\"%g 0 0 0 %g 0 0 0 %g\" Properties=numbers:I:1:pos:R:3:mass:R:1\n',box_length);

for n =1 : N
    fprintf(fid, '%d %g %g %g %g\n', 0, r(n, :), 28);
end
fclose(fid);
