close all
clc;

load a.txt
n=length(a);
lattice_constant=5.46863;
nxyz=10;
max_neighbor_num=300;
cut=10;
atom_num_singer=8;
atom_num=8000;
atom_mass=28;


for ii=1:1:n
    item=1;
    a0=a(ii);
    b0=a(ii);
    c0=a(ii);  
    r0 = [0.0, 0.0, 0.5, 0.5, 0.25, 0.25, 0.75, 0.75; ...
        0.0, 0.5, 0.0, 0.5, 0.25, 0.75, 0.25, 0.75; ...
        0.0, 0.5, 0.5, 0.0, 0.25, 0.75, 0.75, 0.25].';
    results=zeros(atom_num,6);
    for i=0:1:nxyz-1
        for j=0:1:nxyz-1
            for k=0:1:nxyz-1
                for q=1:1:atom_num_singer
                    results(item,2)=r0(q,1)*a0+i*a0;
                    results(item,3)=r0(q,2)*b0+j*b0;
                    results(item,4)=r0(q,3)*c0+k*c0;
                    results(item,5)=atom_mass;
                    results(item,6)=item-1;
                    item=item+1;
                end
            end
        end
    end
    mkdir('.\Si_xyz', num2str(ii));
    str= num2str(ii);
    main_path=cd;
    fid=fopen(['.\Si_xyz\',str ,'\xyz.in'],'w');
    fprintf(fid,'%d %d %d %d %d %d\n',atom_num,max_neighbor_num,cut,0,0,1');
    fprintf(fid,'%d %d %d %g %g %g\n',1,1,1,a0*nxyz,b0*nxyz,c0*nxyz');
    
    
    fprintf(fid,'%d %g %g %g %g %g\n',results');
    
    fclose(fid);

     
    copyfile('run.in', ['.\Si_xyz\',str ]);    
  

    
end

for ii=n+1:1:2*n
    item=1;
    a0=a(ii-n);  
    b0=a(ii-n);  
    c0=lattice_constant;

    r0 = [0.0, 0.0, 0.5, 0.5, 0.25, 0.25, 0.75, 0.75; ...
        0.0, 0.5, 0.0, 0.5, 0.25, 0.75, 0.25, 0.75; ...
        0.0, 0.5, 0.5, 0.0, 0.25, 0.75, 0.75, 0.25].';
    results=zeros(atom_num,6);
    for i=0:1:nxyz-1
        for j=0:1:nxyz-1
            for k=0:1:nxyz-1
                for q=1:1:atom_num_singer
                    results(item,2)=r0(q,1)*a0+i*a0;
                    results(item,3)=r0(q,2)*b0+j*b0;
                    results(item,4)=r0(q,3)*c0+k*c0;
                    results(item,5)=atom_mass;
                    results(item,6)=item-1;
                    item=item+1;
                end
            end
        end
    end
    mkdir('.\Si_xyz', num2str(ii));
    str= num2str(ii);
    main_path=cd;
    fid=fopen(['.\Si_xyz\',str ,'\xyz.in'],'w');
    fprintf(fid,'%d %d %d %d %d %d\n',atom_num,max_neighbor_num,cut,0,0,1');
    fprintf(fid,'%d %d %d %g %g %g\n',1,1,1,a0*nxyz,b0*nxyz,c0*nxyz');
    
    
    fprintf(fid,'%d %g %g %g %d %d\n',results');
    
    fclose(fid);
   
     
    copyfile('run.in', ['.\Si_xyz\',str ]);    
  

    
end

for ii=2*n+1:1:3*n
    item=1;
    a0=a(ii-2*n);  
    b0=lattice_constant;
    c0=lattice_constant;
    r0 = [0.0, 0.0, 0.5, 0.5, 0.25, 0.25, 0.75, 0.75; ...
        0.0, 0.5, 0.0, 0.5, 0.25, 0.75, 0.25, 0.75; ...
        0.0, 0.5, 0.5, 0.0, 0.25, 0.75, 0.75, 0.25].';
    results=zeros(atom_num,6);
    for i=0:1:nxyz-1
        for j=0:1:nxyz-1
            for k=0:1:nxyz-1
                for q=1:1:atom_num_singer
                    results(item,2)=r0(q,1)*a0+i*a0;
                    results(item,3)=r0(q,2)*b0+j*b0;
                    results(item,4)=r0(q,3)*c0+k*c0;
                    results(item,5)=atom_mass;
                    results(item,6)=item-1;
                    item=item+1;
                end
            end
        end
    end
    mkdir('.\Si_xyz', num2str(ii));
    str= num2str(ii);
    main_path=cd;
    fid=fopen(['.\Si_xyz\',str ,'\xyz.in'],'w');
    fprintf(fid,'%d %d %d %d %d %d\n',atom_num,max_neighbor_num,cut,0,0,1');
    fprintf(fid,'%d %d %d %g %g %g\n',1,1,1,a0*nxyz,b0*nxyz,c0*nxyz');
    fprintf(fid,'%d %g %g %g %d %d\n',results');
    fclose(fid);
         
    copyfile('run.in', ['.\Si_xyz\',str ]);    
  

end

fid=fopen(['.\Si_xyz\','input.txt'],'w');
fprintf(fid,'%d  \r\n',3*n);
for i=1:1:3*n
    fprintf(fid,'examples/gpumd_lattice/energy&stress/Si_xyz/Si_xyz/%d  \n',i);
end


fclose(fid);























