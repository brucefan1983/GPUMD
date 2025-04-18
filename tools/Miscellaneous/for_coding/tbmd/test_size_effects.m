clear; close all;
Ne=4;
energy_diamond=zeros(Ne,1);
N_diamond=zeros(Ne,1);

para=[-2.99, 3.71, -5.0, 4.7, 5.5, -1.55];

tic;
for n=1:Ne
[N_diamond(n),L,r]=find_diamond(n,1.55);
[energy_diamond(n)]=find_force(N_diamond(n),L,r,para);
end
toc

figure;
plot(1:4,energy_diamond./N_diamond,'s-','linewidth',2);hold on;
xlabel('supercell size');
ylabel('energy (eV/atom)');
set(gca,'fontsize',15);
