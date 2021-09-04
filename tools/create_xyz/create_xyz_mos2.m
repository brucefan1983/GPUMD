clear; close all;

r0=[1/2,0,0;0,1/6,1;0,1/6,-1;0,1/2,0;1/2,2/3,1;1/2,2/3,-1];
t0=[0;1;1;0;1;1];
n0=size(r0,1);
nxyz=[10,6,1];
N=nxyz(1)*nxyz(2)*nxyz(3)*n0;
a=[3.16,3.16*sqrt(3),1.62];
b=[3.16,3.16*sqrt(3),6.15];
r=zeros(N,3);
t=zeros(N,1);
label=zeros(N,1);

n=0;
for ny=0:nxyz(2)-1
    for nx=0:nxyz(1)-1
        for nz=0:nxyz(3)-1
            for m=1:n0
                n=n+1;
                r(n,:)=a.*([nx,ny,nz]+r0(m,:));
                t(n)=t0(m);
                label(n)=0;
            end
        end
    end
end

fid=fopen('xyz.in','w');
fprintf(fid,'%d %g %g 0 0 0\n',N,500,11.5);
fprintf(fid,'%d %d %d %g %g %g\n',1,1,0,b.*nxyz);
for n=1:N
    if t(n)==0
        fprintf(fid,'%d %g %g %g %g\n',t(n),r(n,:),96);
    else
        fprintf(fid,'%d %g %g %g %g\n',t(n),r(n,:),32);
    end
end
fclose(fid);
