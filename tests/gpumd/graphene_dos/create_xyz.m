clear; close all;

r0=[1/2,0,0;0,1/6,0;0,1/2,0;1/2,2/3,0];
n0=size(r0,1);
nxyz=[60,36,1];
N=nxyz(1)*nxyz(2)*nxyz(3)*n0;
a=[1.44*sqrt(3),1.44*3,3.35];

r=zeros(N,3);
label=zeros(N,1);
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

fid=fopen('model.xyz','w');
fprintf(fid,'%d\n',N);
fprintf(fid,'triclinic=f pbc=\"T T F\" Lattice=\"%g 0 0 0 %g 0 0 0 %g\" Properties=numbers:I:1:pos:R:3:mass:R:1\n',a.*nxyz);
for n=1:N
    fprintf(fid,'%d %g %g %g %g\n',0,r(n,:),12);
end
fclose(fid);
