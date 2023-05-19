function [K,G] = get_K_and_G(file_shc,Nc,Nw,deltaT,vol) 

data=load(file_shc);

Nt=Nc*2-1;
N_one_run=Nt+Nw;
N_total=size(data,1);
N_runs=N_total/N_one_run;
data_ave=zeros(N_one_run,3);
for n=1:N_runs
    index=(n-1)*N_one_run+1:n*N_one_run;
    data_ave=data_ave+data(index,:);
end
data_ave=data_ave/N_runs;

K=zeros(Nt,2);
K(:,1)=data_ave(1:Nt,1); % ps
K(:,2)=1.60217663e4*sum(data_ave(1:Nt,2:3),2)/vol; % W/m^2
G=zeros(Nw,2);
G(:,1)=data_ave(Nt+1:end,1)/2/pi; % THz
G(:,2)=1.60217663e4*sum(data_ave(Nt+1:end,2:3),2)/vol/deltaT; % GW/m^2/K/THz
