clear; close all;
load omega2.out; % output from GPUMD

% parameters used for GPUMD:
a0=5.431; % lattice constant
a=[0 1 1;1 0 1;1 1 0]*a0/2;
Nb=2;
special_k=[0,0,0;1/2,0,1/2;3/8,3/8,3/4;0,0,0;1/2,1/2,1/2];
name_special_k={'$\Gamma$','X','K','$\Gamma$','L'};
Nk=100; % number of k points between two special ones

% get the k points
[K,k_distance]=find_k(Nk,special_k.',a);

% get the frequencies (one can check if there are imaginary frequencies)
nu=real(sqrt(omega2.'))/2/pi; % from omega^2 to nu

% plot the phonon dispersion
figure;
k_distance=k_distance/(2*pi/a0);
max_nu=max(max(nu));
plot(ones(100,1)*k_distance(1),linspace(0,max_nu*1.1,100),'k-','linewidth',2);
hold on;
for n=1:size(name_special_k,2)-1
    plot(linspace(k_distance(n),k_distance(n+1),Nk),nu(:,(n-1)*Nk+1:n*Nk),'r.');
    plot(ones(100,1)*k_distance(n+1),linspace(0,max_nu*1.1,100),'k-','linewidth',2);
end
set(gca,'xtick',[],'fontsize',12);
ylabel('\nu (THz)','fontsize',12);
axis tight;
for n=1:size(name_special_k,2)
    text(k_distance(n),-max_nu*0.05,name_special_k(n),...
        'interpreter','latex','fontsize',12);
end
