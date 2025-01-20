function [energy, force] = find_force_train(N, box, r, para)
    rc=4;
    [NN,NL]=find_neighbor(N,box,rc,r);
    %[energy_repulsive,force_repulsive]=find_force_repulsive(N, NN, NL, box, r);
    [energy_band,force_band]=find_force_band_train(N, NN, NL, box, rc, r, para);
    % energy=energy_repulsive+energy_band;
    % force=force_repulsive+force_band;
    energy=energy_band;
    force=force_band;
end
