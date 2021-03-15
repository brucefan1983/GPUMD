function [r, box] = find_r0_box0()
r0 = [0.0, 0.0, 0.5, 0.5; ...
      0.0, 0.5, 0.0, 0.5; ...
      0.0, 0.5, 0.5, 0.0].';
n0 = size(r0, 1);  
nx = 4;
a = 5.6;
box = eye(3) * a * nx;

r = zeros(n0*nx*nx*nx, 3);
n = 0;
for ix = 0 : nx - 1
    for iy = 0 : nx - 1
        for iz = 0 : nx - 1
            for m = 1 : n0
                n = n + 1;
                r(n, :) = a * ([ix, iy, iz] + r0(m, :));   
            end
        end
    end
end