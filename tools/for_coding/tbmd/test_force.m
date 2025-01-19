clear; close all;

[N,L,r]=find_graphene(1.42);
r=r+rand(N,3)*0.1;
[NN,NL]=find_neighbor(N,L,[1 1 1],3,r);
tic
[energy,f_analytical]=find_force(N,3,NN,NL,L,[1 1 1],r);
toc

f_finite=zeros(N,3);
delta=1e-5;
for n=1:N
    tic;
    for d=1:3
    rpx=r;rpx(n,d)=rpx(n,d)+delta;
    rmx=r;rmx(n,d)=rmx(n,d)-delta;
    [ep]=find_force(N,3,NN,NL,L,[1 1 1],rpx);
    [em]=find_force(N,3,NN,NL,L,[1 1 1],rmx);
    f_finite(n,d)=(em-ep)/(2*delta);
    end
    toc
end

figure;
plot(f_finite(:,1)-f_analytical(:,1),'d','linewidth',2);hold on;
plot(f_finite(:,2)-f_analytical(:,2),'s','linewidth',2);hold on;
plot(f_finite(:,3)-f_analytical(:,3),'o','linewidth',2);hold on;
xlabel('Atom index');
ylabel('Force difference (eV/A)');
legend('x','y','z');
set(gca,'fontsize',15);
