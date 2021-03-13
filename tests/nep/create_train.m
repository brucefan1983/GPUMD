clear; close all;
[r0, box0] = find_r0_box0();
Na = size(r0, 1); % number of atoms in one configuration
Nc = 30; % number of configurations
fid = fopen('train.in', 'w');
fprintf(fid,'%d\n',Nc);
for nc=1:(Nc)
    fprintf(fid,'%d\n',Na);
end
energy = zeros(1, Nc);
virial = zeros(6, Nc);
for nc = 1 : Nc
    r = r0 + rand(Na, 3) * 0.01*nc;
    box = box0;
    [energy(nc), virial(:,nc), force] = find_E(r, box);
    fprintf(fid,'%f %f %f %f %f %f %f\n',energy(nc), virial(:,nc));
    fprintf(fid,'%f %f %f %f %f %f %f %f %f\n',box);
    for i = 1 : Na
        fprintf(fid,'0 %f %f %f %f %f %f\n',r(i,:), force(i,:));
    end

end