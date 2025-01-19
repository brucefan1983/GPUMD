clear; close all;

%para=[-2.99, 3.71, -5.0, 4.7, 5.5, -1.55];

para=[36.6033   -2.4823    6.9762   -1.0257    0.6155   -3.7350    0.0001   -0.1501    1.2124    0.9330    0.9801 ...
   -2.3740    1.8135   -2.6724   -0.7851    1.4132   -0.7968    2.4137    0.2738  -11.5215   -1.6558   -1.1167];


[N,L,r]=find_graphene(1.42);
r=r+rand(N,3)*0.1;
[NN,NL]=find_neighbor(N,L,3,r);
tic
[energy,f_analytical]=find_force_train(N,NN,NL,L,r,para);
toc

f_finite=zeros(N,3);
delta=2e-5;
for n=1:N
    tic;
    for d=1:3
    rpx=r;rpx(n,d)=rpx(n,d)+delta;
    rmx=r;rmx(n,d)=rmx(n,d)-delta;
    [ep]=find_force_train(N,NN,NL,L,rpx,para);
    [em]=find_force_train(N,NN,NL,L,rmx,para);
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

figure;
plot(f_analytical(:,:), f_finite(:,:),'.','linewidth',2,'MarkerSize',20);hold on;
plot(linspace(-10,10,100),linspace(-10,10,100));
xlabel('analytical (eV/A)');
ylabel('finite difference (eV/A)');
legend('x','y','z');
set(gca,'fontsize',15);

