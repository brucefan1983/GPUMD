clear; close all;
Ne=4;
energy_diamond=zeros(Ne,1);
N_diamond=zeros(Ne,1);

tic;
for n=1:Ne
[N_diamond(n),L,r]=find_diamond(n,1.4);
[NN,NL]=find_neighbor(N_diamond(n),L,[1 1 1],3,r);
[energy_diamond(n)]=find_force(N_diamond(n),3,NN,NL,L,[1 1 1],r);
end
toc

figure;
plot(1:4,energy_diamond./N_diamond,'s-','linewidth',2);hold on;
xlabel('supercell size');
ylabel('energy (eV/atom)');
legend('diamond','graphene');
set(gca,'fontsize',20);
