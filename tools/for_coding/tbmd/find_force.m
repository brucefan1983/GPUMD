function [energy, force] = find_force(N, box, r, para)
    [NN,NL]=find_neighbor(N,box,3,r);
    [energy_repulsive,force_repulsive]=find_force_repulsive(N, NN, NL, box, r);
    [energy_band,force_band]=find_force_band(N, NN, NL, box, r, para);
    energy=energy_repulsive+energy_band;
    force=force_repulsive+force_band;
end
