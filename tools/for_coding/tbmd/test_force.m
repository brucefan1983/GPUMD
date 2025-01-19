clear; close all;

%para=[-2.99, 3.71, -5.0, 4.7, 5.5, -1.55];


  para=[
   -5.6447   -0.3769   -1.0889   -0.9996   -0.6293   -3.8991    0.3825    0.6056    1.4281   -0.6272    1.2932   -1.5815    1.0794...
    2.1956    1.2200    3.8104   -2.0141    0.8805    2.1493    1.1209   -3.5681   -0.8650   -3.2628    1.7338   -2.0385   -1.9228...
   -0.6791    0.2370    6.5588    1.6910   -0.2727   -0.9297    1.4523    3.3451    2.0848   -3.6105    3.7906    1.3693   -0.2695...
   -1.1855    0.8958    1.7504    0.4297    1.0103    1.5160   -1.2285   -0.3290   -1.6213    4.9754    2.1861    1.6752    2.0441...
   -0.0034    2.8440    1.4665    2.3616    1.4316    1.0942   -3.4234   -0.2475    3.5347    2.9264    1.4146   -1.2184    2.1778...
    0.1930
];


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

