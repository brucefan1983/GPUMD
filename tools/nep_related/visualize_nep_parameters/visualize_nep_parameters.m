function []=visualize_nep_parameters(nep_file,N_type,basis_size,n_max,l_max,neuron)

nep=load(nep_file); nep=nep(:,1);
basis_size_radial=basis_size(1);
basis_size_angular=basis_size(2);
n_max_radial=n_max(1);
n_max_angular=n_max(2);
l_max=4+sum(l_max(2:end)~=0);%four 3-body + one 4-boby (if any) + one 5-body (if any)

% calculated numbers
dim_radial = n_max_radial + 1
dim_angular = (n_max_angular + 1) * l_max
dim = dim_radial + dim_angular
N_ann = (dim + 2) * neuron*N_type + 1

% neural network parameters
para_ann=nep(1:N_ann);
for n=1:N_type
    offset=(n-1)*(dim + 2) * neuron;
    w0=para_ann(offset+1:offset+neuron*dim);
    w0=reshape(w0,dim,neuron);
    figure;
    plot(abs(w0).','.-','markersize',30)
    xlabel('neuron index');
    ylabel('absolute connection weight value');
    set(gca,'fontsize',15);
end

% descriptor parameters
para_c=nep(N_ann+1:end);
para_c=reshape(para_c,N_type*N_type,size(para_c,1)/(N_type*N_type));
para_c_radial=para_c(:,1:(n_max_radial+1)*(basis_size_radial+1));
para_c_angular=para_c(:,(n_max_radial+1)*(basis_size_radial+1)+1:end);

% radial part with g_n in x
figure;
for n=1:N_type*N_type
    subplot(N_type,N_type,n);
    plot(abs(reshape(para_c_radial(n,:),basis_size_radial+1,n_max_radial+1)).','.-','markersize',30)
    ylim([0,max(max(abs(para_c_radial)))]);
    xlabel('g_{n} index');
    ylabel('absolute c_{nk} value');
    set(gca,'fontsize',15);
end

% radial part with f_k in x
figure;
for n=1:N_type*N_type
    subplot(N_type,N_type,n);
    plot(abs(reshape(para_c_radial(n,:),basis_size_radial+1,n_max_radial+1)),'.-','markersize',30)
    ylim([0,max(max(abs(para_c_radial)))]);
    xlabel('f_{k} index');
    ylabel('absolute c_{nk} value');
    set(gca,'fontsize',15);
end

% angular part with g_n in x
figure;
for n=1:N_type*N_type
    subplot(N_type,N_type,n);
    plot(abs(reshape(para_c_angular(n,:),basis_size_angular+1,n_max_angular+1)).','.-','markersize',30)
    ylim([0,max(max(abs(para_c_angular)))]);
    xlabel('g_{n} index');
    ylabel('absolute c_{nk} value');
    set(gca,'fontsize',15);
end

% angular part with f_k in x
figure;
for n=1:N_type*N_type
    subplot(N_type,N_type,n);
    plot(abs(reshape(para_c_angular(n,:),basis_size_angular+1,n_max_angular+1)),'.-','markersize',30)
    ylim([0,max(max(abs(para_c_angular)))]);
    xlabel('f_{k} index');
    ylabel('absolute c_{nk} value');
    set(gca,'fontsize',15);
end
