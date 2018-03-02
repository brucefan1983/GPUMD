clear; close all;
r0 = [0.0, 0.0, 0.5, 0.5, 0.25, 0.25, 0.75, 0.75; ...
      0.0, 0.5, 0.0, 0.5, 0.25, 0.75, 0.25, 0.75; ...
      0.0, 0.5, 0.5, 0.0, 0.25, 0.75, 0.75, 0.25].';
n0 = size(r0, 1);
nxyz = 10 * [1, 1, 1];
N = nxyz(1) * nxyz(2) * nxyz(3) * n0;
a = 4.357 * [1, 1, 1];
box_length = a .* nxyz;
mass0=[28, 28, 28, 28, 12, 12, 12, 12];
type0=[0, 0 , 0, 0, 1, 1, 1, 1];

r = zeros(N, 3);
mass = zeros(N, 1);
type = zeros(N, 1);
n = 0;
for nx = 0 : nxyz(1) - 1
    for ny = 0 : nxyz(2) - 1
        for nz = 0 : nxyz(3) - 1
            for m = 1 : n0
                n = n + 1;
                r(n, :) = a .* ([nx,ny,nz] + r0(m, :));  
                mass(n) = mass0(m);
                type(n) = type0(m);
            end
        end
    end
end

fid = fopen('xyz.in', 'w');
fprintf(fid, '%d %d %g\n', N, 300, 8.35);
fprintf(fid, '%d %d %d %g %g %g\n', 1, 1, 1, box_length);
for n =1 : N
    fprintf(fid, '%d %d %g %g %g %g\n', type(n), 0, mass(n), r(n, :));
end
fclose(fid);


figure;
plot3(r(type==0,1),r(type==0,2), r(type==0,3), 'ro');
hold on;
plot3(r(type==1,1),r(type==1,2), r(type==1,3), 'bo');
axis equal;




