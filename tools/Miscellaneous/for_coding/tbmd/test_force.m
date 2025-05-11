clear; close all;

%para=[-2.99, 3.71, -5.0, 4.7, 5.5, -1.55];

load para;

[N,L,r]=find_graphene(1.42);
r=r+rand(N,3)*0.1;
tic
[energy,f_analytical]=find_force_train(N,L,r,para);
toc

f_finite=zeros(N,3);
delta=2e-5;
for n=1:N
    tic;
    for d=1:3
    rpx=r;rpx(n,d)=rpx(n,d)+delta;
    rmx=r;rmx(n,d)=rmx(n,d)-delta;
    [ep]=find_force_train(N,L,rpx,para);
    [em]=find_force_train(N,L,rmx,para);
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

