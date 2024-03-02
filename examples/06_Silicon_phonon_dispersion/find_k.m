function [K,k_distance]=find_k(Nk,special_k,a)
b=zeros(3,3);
omega=dot(a(:,1),cross(a(:,2),a(:,3)));
b(:,1)=2*pi*cross(a(:,2),a(:,3))/omega;
b(:,2)=2*pi*cross(a(:,3),a(:,1))/omega;
b(:,3)=2*pi*cross(a(:,1),a(:,2))/omega;
num_of_special_k=size(special_k,2);
for n=1:num_of_special_k
    special_k(:,n)=b*special_k(:,n);
end
NK=Nk*num_of_special_k/2;
K=zeros(3,NK);
for n=1:num_of_special_k/2
    for d=1:3
       K(d,(n-1)*Nk+1:n*Nk)=linspace(special_k(d,2*n-1),special_k(d,2*n),Nk);
    end
end

k_distance=zeros(num_of_special_k/2+1,1);
k_distance(1)=0;
for n=1:num_of_special_k/2
    k_distance(n+1)=norm(special_k(:,2*n)-special_k(:,2*n-1));
end
k_distance=cumsum(k_distance);
