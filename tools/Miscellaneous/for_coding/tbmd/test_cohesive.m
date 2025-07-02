clear; close all;
Ne=30;
bond_diamond=linspace(1.27,1.7,Ne);
bond_graphene=linspace(1.25,1.6,Ne);
energy_diamond=zeros(Ne,1);
energy_graphene=zeros(Ne,1);

para=[-2.99, 3.71, -5.0, 4.7, 5.5, -1.55];

tic;
for n=1:Ne
[N_diamond,L,r]=find_diamond(2,bond_diamond(n));
[energy_diamond(n)]=find_force(N_diamond,L,r,para);
[N_graphene,L,r]=find_graphene(bond_graphene(n));
[energy_graphene(n)]=find_force(N_graphene,L,r,para);
end
toc

figure;
plot(bond_diamond,energy_diamond/N_diamond,'s-','linewidth',2);hold on;
plot(bond_graphene,energy_graphene/N_graphene,'o-','linewidth',2);
xlabel('bond length (A)');
ylabel('energy (eV/atom)');
legend('diamond','graphene');
set(gca,'fontsize',15);
