function [loss]=snes(x0,y0,N_par,N_gen)
N_pop=50;
loss=ones(N_gen,1);
m=rand(1,N_par)-0.5;
s=0.1*ones(1,N_par);
eta=[1,(3+log(N_par))/(5*sqrt(N_par))/2];
u=max(0,log(N_pop/2+1)-log(1:N_pop));
u=u/sum(u)-1/N_pop;
for gen=1:N_gen
    r=randn(N_pop,N_par);
    pop=repmat(m,N_pop,1)+repmat(s,N_pop,1).*r;
    loss_all=ann(x0,y0,pop);
    [loss_all,index]=sort(loss_all);
    r=r(index,:);
    loss(gen)=loss_all(1);
    m=m+eta(1)*s.*(u*r);
    s=s.*exp(eta(2)*(u*(r.*r-1)));
end
