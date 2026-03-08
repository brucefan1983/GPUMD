clear; close all;

% old file
load para.txt;

% inputs
num_species=89;
neuron=80;
n_max=[4,4];
l_max=[4,2,0];
scaling_factor=2;

% calculated
num_angular=l_max(1)+(l_max(2)>0)+(l_max(3)>0)
dim=(n_max(1)+1) + (n_max(2)+1)*num_angular
num_ann_para_1=(dim+1)*neuron
num_ann_para_2=neuron
num_total=size(para,1)

% new file
fid=fopen('nep_new.txt','w');

offset=1;
for n=1:num_species
    for m=1:num_ann_para_1
        fprintf(fid,'%15.7e\n',para(offset));
        offset=offset+1
    end 
    for m=1:num_ann_para_2
        fprintf(fid,'%15.7e\n',para(offset)*scaling_factor^3);
        offset=offset+1
    end 
end
fprintf(fid,'%15.7e\n',para(offset)*scaling_factor^3);
offset=offset+1

for n=offset:num_total
    fprintf(fid,"%15.7e\n",para(n));
    offset=offset+1
end

fclose(fid);

