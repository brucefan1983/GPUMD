lc=3.1854;
nxyz=100;
p=0.0;
r0 = [0.0, 0.5; ...
    0.0, 0.5,; ...
    0.0, 0.5].';
atom_num=size(r0,1);
n = nxyz * nxyz * nxyz * atom_num;
fid=fopen('.\xyz.in','w');
fprintf(fid,'%d 500 6 1 0 0\n',n);
a0=lc;
b0=lc;
c0=lc;
fprintf(fid,'1 1 1 %g 0 0 0 %g 0 0 0 %g\n',a0*nxyz,b0*nxyz,c0*nxyz');
for nx=0:1:nxyz-1
    for ny=0:1:nxyz-1
        for nz=0:1:nxyz-1
            R=(2*rand(2,3)-1).*p;
            fprintf(fid,'W %g %g %g 184\n',r0(1,1)*a0+nx*a0+R(1,1) ,r0(1,2)*b0+ny*b0+R(1,2) ,r0(1,3)*c0+nz*c0+R(1,3));
            fprintf(fid,'W %g %g %g 184\n',r0(2,1)*a0+nx*a0+R(2,1),r0(2,2)*b0+ny*b0++R(2,2),r0(2,3)*c0+nz*c0++R(2,3));
        end
    end
end