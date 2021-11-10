clear; close all;
r0 = [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0; ...
    0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0; ...
    0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5].';
label0=[0 0 0 0 1 1 1 1];
type0 = [0 0 0 0 1 1 1 1];
mass0 = [128 128 128 128 207 207 207 207];
n0 = size(r0, 1);
nxyz = 10 * [1, 1, 1];
N = nxyz(1) * nxyz(2) * nxyz(3) * n0;
a = 6.5704 * [1, 1, 1];
box_length = a .* nxyz;

r = zeros(N, 3);
type = zeros(N, 1);
mass = zeros(N, 1);
label = zeros(N, 1);
n = 0;
for nx = 0 : nxyz(1) - 1
    for ny = 0 : nxyz(2) - 1
        for nz = 0 : nxyz(3) - 1
            for m = 1 : n0
                n = n + 1;
                r(n, :) = a .* ([nx,ny,nz] + r0(m, :));
                type(n) = type0(m);
                mass(n) = mass0(m);
                label(n) = label0(m);
            end
        end
    end
end

fid = fopen('xyz.in', 'w');
fprintf(fid, '%d %d %f 0 0 0\n', N, 100, 9.0);
fprintf(fid, '%d %d %d %f %f %f\n', 1, 1, 1, box_length);

for n =1 : N
    if type(n)==0
        fprintf(fid, 'Te %f %f %f %f\n', r(n, :), mass(n));
    else
        fprintf(fid, 'Pb %f %f %f %f\n', r(n, :), mass(n));
    end
end
fclose(fid);

% create basis.in
fid=fopen('basis.in','w');
fprintf(fid,'%d\n', 2);
fprintf(fid,'0 128\n');
fprintf(fid,'4 207\n');
for n=1:N
    fprintf(fid,'%d\n', label(n));
end
fclose(fid);

% create kpoints.in
primitive_cell=[0 1 1;1 0 1;1 1 0]*a(1)/2;
special_k=[0,0,0;    1/2,0,1/2;    % Gamma -> X
    1/2,0,1/2;       5/8,1/4,5/8;  % X -> U=K
    3/8,3/8,3/4;     0,0,0;        % K -> Gamma
    0,0,0;           1/2,1/2,1/2]; % Gamma -> L
Nk=100; % number of k points between two special ones
[K,k_norm]=find_k(Nk, special_k.', primitive_cell);
fid=fopen('kpoints.in','w');
fprintf(fid,'%d\n', size(K,2));
for n=1:size(K,2)
    fprintf(fid,'%f %f %f\n', K(:,n));
end
fclose(fid);


