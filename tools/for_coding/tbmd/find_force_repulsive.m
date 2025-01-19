function [energy, force] = find_force_repulsive(N, neighbor_number, neighbor_list, box, r)
    D=3; energy = 0; force = zeros(N, D);
    sum_of_phi = zeros(N, 1);
    for n1 = 1 : N
        for k = 1 : neighbor_number(n1)
            n2 = neighbor_list(n1, k);
            r12 = r(n2, :) - r(n1, :);
            r12 = r12 - round(r12./box).*box; % minimum image convention
            d12 = sqrt(sum(r12.*r12));
            sum_of_phi(n1) = sum_of_phi(n1) + phi(d12);
        end
    end
    for n1 = 1 : N
        for k = 1 : neighbor_number(n1)
            n2 = neighbor_list(n1, k);
            r12 = r(n2, :) - r(n1, :);
            r12 = r12 - round(r12./box).*box; % minimum image convention
            d12 = sqrt(sum(r12.*r12));
            temp = phi_d(d12)*(f_d(sum_of_phi(n1))+f_d(sum_of_phi(n2)))/d12;
            force(n1, :) = force(n1, :) + r12 * temp;
        end
        energy = energy + f(sum_of_phi(n1));
    end
end

function y = f(x)
    c0=-2.5909765118191; 
    c1=0.5721151498619; 
    c2=-1.7896349903996e-3;
    c3=2.3539221516757e-5; 
    c4=-1.2425116955159e-7;
    y=c0+x*(c1+x*(c2+x*(c3+x*c4)));
end
function y = f_d(x)
    c1=0.5721151498619; 
    c2=-1.7896349903996e-3;
    c3=2.3539221516757e-5; 
    c4=-1.2425116955159e-7;
    y=c1+x*(2*c2+x*(3*c3+x*4*c4));
end
function y = phi(r)
	phi0=8.18555;
	m=3.30304;
    mc=8.6655;
	dc=2.1052;
	d0=1.64;
    y=phi0*(d0/r)^m*exp(m*(-(r/dc)^mc+(d0/dc)^mc));
end
function y = phi_d(r)
	m=3.30304;
    mc=8.6655;
	dc=2.1052;
    y=-m*phi(r)*(1+mc*(r/dc)^mc)/r;
end

