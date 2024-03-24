clear; close all;

% inputs
N_elements_1 = 6;
N_elements_2 = 6;
nep1=load("12/nep.restart");
nep2=load("13/nep.restart");
n_max= [4 4];
basis_size =[8 8];
num_L = 5; % 3body + 4body
neuron= 30;

N_elements_12=N_elements_1+N_elements_2;
N_des = n_max(1)+1 + (n_max(2)+1) * num_L;
N_ann = (N_des + 2) * neuron;
N_c_radial = (n_max(1)+1) * (basis_size(1)+1);
N_c_angular = (n_max(2)+1) * (basis_size(2)+1);
N_c=N_c_radial+N_c_angular;
N_1 = N_ann * N_elements_1 + 1 + N_c * N_elements_1 * N_elements_1;
N_2 = N_ann * N_elements_2 + 1 + N_c * N_elements_2 * N_elements_2;
N_12 = N_ann * N_elements_12 + 1 + N_c * N_elements_12 * N_elements_12;

fid=fopen("nep.restart","w");
% ANN for elements in model 1
for m=1:N_ann*N_elements_1
    fprintf(fid,"%15.7e %15.7e\n",nep1(m,:));
end
% ANN for elements in model 2
for m=1:N_ann*N_elements_2
    fprintf(fid,"%15.7e %15.7e\n",nep2(m,:));
end

% The global bias is taken as 0
fprintf(fid,"%15.7e %15.7e\n",0,0);

bias_model_1 = nep1(N_ann*N_elements_1+1)
bias_model_2 = nep2(N_ann*N_elements_2+1)

offset1=N_ann*N_elements_1+1+1;
offset2=N_ann*N_elements_2+1+1;
for m=1:N_c
    for n1=1:N_elements_12
        for n2=1:N_elements_12
            if n1>=1 && n1<=N_elements_1 && n2>=1 && n2<=N_elements_1
                fprintf(fid,"%15.7e %15.7e\n",nep1(offset1,:));
                offset1=offset1+1;
            elseif n1>=(N_elements_1+1) && n1<=N_elements_12 && n2>=(N_elements_1+1) && n2<=N_elements_12
                fprintf(fid,"%15.7e %15.7e\n",nep2(offset2,:));
                offset2=offset2+1;
            else
                fprintf(fid,"%15.7e %15.7e\n",rand-0.5,0.1);
            end
        end
    end
end

fclose(fid);

