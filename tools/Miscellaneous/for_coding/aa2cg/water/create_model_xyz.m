N=1536;
L=[24.83935093523772 0.0 0.0 0.0 24.83935093523772 0.0 0.0 0.0 24.83935093523772];
load aa.xyz;

elements=['O','H','H'];

fid_model_xyz=fopen('model.xyz','w');
fid_bead_name=fopen('bead_name.txt','w');

fprintf(fid_model_xyz, "%d\n",N);
fprintf(fid_model_xyz, 'Lattice=\"24.83935093523772 0.0 0.0 0.0 24.83935093523772 0.0 0.0 0.0 24.83935093523772\" Properties=Species:S:1:pos:R:3:group:I:1\n');

count=0;
for m=1:N/3
    is_bead = rand<0.9;

    for k=1:3
        fprintf(fid_model_xyz, "%s %g, %g, %g %d\n", elements(k), aa((m-1)*3+k,1),aa((m-1)*3+k,2),aa((m-1)*3+k,3),count);
        if ~is_bead
            fprintf(fid_bead_name, "%s\n", elements(k));
        else
            fprintf(fid_bead_name, "F\n");
        end
        if ~is_bead
            count=count+1;
        end
    end
    if is_bead
        count=count+1;
    end
end

fclose(fid_model_xyz);
fclose(fid_bead_name);