function [nu]=find_nu(D,Nb,NK)
nu=zeros(3*Nb,NK);
for nk=1:NK
    DR=D((nk-1)*3*Nb+1:nk*3*Nb,1:end/2);
    DI=D((nk-1)*3*Nb+1:nk*3*Nb,end/2+1:end);
    DK=DR+1i*DI;
    nu(:,nk)=real(sqrt(eig(DK)));
end
nu=nu*1000/10.18/2/pi; % in units of THz now