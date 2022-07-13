clear; close all;

r0=[1/2,0,0;0,1/6,0;0,1/2,0;1/2,2/3,0];
n0=size(r0,1);
nxyz=[100,60,1];
N=nxyz(1)*nxyz(2)*nxyz(3)*n0;
a=[1.42*sqrt(3),1.42*3,3.35];
r=zeros(N,3);
label=zeros(N,1);

n=0;
for ny=0:nxyz(2)-1
    for nx=0:nxyz(1)-1
        for nz=0:nxyz(3)-1
            for m=1:n0
                n=n+1;
                r(n,:)=a.*([nx,ny,nz]+r0(m,:));
                if ny<10
                    label(n)=0;
                else
                    label(n)=1;
                end
            end
        end
    end
end

fid=fopen('xyz.in','w');
fprintf(fid,'%d 0 0 1\n',N);
fprintf(fid,'%d %d %d %g %g %g\n',1,1,0,a.*nxyz);
for n=1:N
    fprintf(fid,'%d %g %g %g %g %d\n',0,r(n,:),12,label(n));
end
fclose(fid);
