clear; close all;

% old file
load nep.restart;

% inputs
num_species=89;
neuron=80;
n_max=[6,4];
l_max=[4,2,0];

% calculated
num_angular=l_max(1)+(l_max(2)>0)+(l_max(3)>0);
dim=(n_max(1)+1) + (n_max(2)+1)*num_angular;
num_ann_para=(dim+2)*neuron;
num_total=size(nep,1);


% new file
fid=fopen('nep_new.restart','w');

offset=1;
for n=1:num_species
    for m=1:num_ann_para
        fprintf(fid,'%15.7e %15.7e\n',nep(offset,:));
        offset=offset+1;
    end
    if n==10
        fprintf(fid,'%15.7e%15.7e\n',-83.3701 +0.0301684,0.01); 
    elseif n==18
        fprintf(fid,'%15.7e%15.7e\n',-83.3701 +0.0587795,0.01); 
    else
        fprintf(fid,'%15.7e%15.7e\n',0,0.01); 
    end
end

for n=offset:num_total
    fprintf(fid,"%15.7e %15.7e\n",nep(n,:));
end

fclose(fid);

