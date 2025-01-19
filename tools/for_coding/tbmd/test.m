clear; close all;
Ne=30;
bond_diamond=linspace(1.27,1.7,Ne);
bond_graphene=linspace(1.25,1.6,Ne);
energy_diamond=zeros(Ne,1);
energy_graphene=zeros(Ne,1);

tic;
for n=1:Ne
[N_diamond,L,r]=find_diamond(bond_diamond(n));
[NN,NL]=find_neighbor(N_diamond,L,[1 1 1],3,r);
[energy_diamond(n)]=find_force(N_diamond,3,NN,NL,L,[1 1 1],r);
[N_graphene,L,r]=find_graphene(bond_graphene(n));
[NN,NL]=find_neighbor(N_graphene,L,[1 1 1],3,r);
[energy_graphene(n)]=find_force(N_graphene,3,NN,NL,L,[1 1 1],r);
end
toc

figure;
plot(bond_diamond,energy_diamond/N_diamond,'s-','linewidth',2);hold on;
plot(bond_graphene,energy_graphene/N_graphene,'o-','linewidth',2);
xlabel('bond length (A)');
ylabel('energy (eV/atom)');
legend('diamond','graphene');
set(gca,'fontsize',20);
