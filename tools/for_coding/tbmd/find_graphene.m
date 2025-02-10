function [N,L,r]=find_graphene(bond_length)

r0=[1/2,0,0;0,1/6,0;0,1/2,0;1/2,2/3,0];
n0=size(r0,1);
nxyz=[5,3,1];
N=nxyz(1)*nxyz(2)*nxyz(3)*n0;
a=[bond_length*sqrt(3),bond_length*3,10];
L=a.*nxyz;
r=zeros(N,3);
n=0;
for nx=0:nxyz(1)-1
    for ny=0:nxyz(2)-1
        for nz=0:nxyz(3)-1
            for m=1:n0
                n=n+1;
                r(n,:)=a.*([nx,ny,nz]+r0(m,:));   
            end
        end
    end
end

