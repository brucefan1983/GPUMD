function [energy, virial, force] = find_E(r, box)
epsilon = 1.032e-2;
epsilon4 = epsilon * 4;
sigma = 3.405;
sigma6 = sigma^6;
sigma12 = sigma6 * sigma6;
rc = 5;
energy_cut = epsilon4 * (sigma12 / rc^12 - sigma6 / rc^6);
d_energy_cut = epsilon4 * (6 * sigma6 / rc^7 - 12 * sigma12 / rc^13);

N = size(r, 1);
energy = 0;
virial = zeros(6, 1);
force = zeros(N, 3);
for n1=1:N-1
    for n2=(n1+1):N
        r12=r(n2,:)-r(n1,:);                  % position difference vector
        r12=r12.';                            % column vector
        r12=box\r12;                          % transform to cubic box
        r12=r12-round(r12);                   % mininum image convention
        r12=box*r12;                          % transform back
        d12=sqrt(sum(r12.*r12));              % distance
        if d12 > rc
            continue;
        end
        
        d12inv2 = 1 / (d12 * d12);
        d12inv6 = d12inv2 * d12inv2 * d12inv2;
        d12inv8 = d12inv2 * d12inv6;
        d12inv12 = d12inv6 * d12inv6;
        d12inv14 = d12inv2 * d12inv12;

        energy = energy + epsilon4 * (sigma12 * d12inv12 - sigma6 * d12inv6) ...
        - energy_cut - d_energy_cut * (d12 - rc);
        f2 = epsilon4 * (6 * sigma6 * d12inv8 - 12 * sigma12 * d12inv14) - d_energy_cut / d12;
        
        force(n1, :) = force(n1, :) + f2 * r12.';
        force(n2, :) = force(n2, :) - f2 * r12.';
        virial(1) = virial(1) - r12(1) * r12(1) * f2;
        virial(2) = virial(2) - r12(2) * r12(2) * f2;
        virial(3) = virial(3) - r12(3) * r12(3) * f2;
        virial(4) = virial(4) - r12(1) * r12(2) * f2;
        virial(5) = virial(5) - r12(2) * r12(3) * f2;
        virial(6) = virial(6) - r12(3) * r12(1) * f2;
    end
end

