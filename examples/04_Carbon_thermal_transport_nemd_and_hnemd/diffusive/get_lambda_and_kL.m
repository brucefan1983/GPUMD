function [lambda,kappa_vs_L]=get_lambda_and_kL(G,kappa,len)
lambda=kappa(:,2)./G(:,2);
kappa_vs_L=zeros(length(len),1);
for l=1:length(len)
    tmp=kappa(:,2)./(1+lambda/len(l));
    kappa_vs_L(l)=sum(tmp)*(G(2,1)-G(1,1));
end