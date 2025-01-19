function [energy, force] = find_force(N, D, neighbor_number, neighbor_list, L, pbc, r)
    N4 = N*4; N2 = N4/2; H0 = zeros(N4, N4); H  = zeros(N4, N4);
    L_times_pbc=L.*pbc;
    energy = 0; force = zeros(N, D);
    on_site_matrix = diag([-2.99, 3.71, 3.71, 3.71]);
    v_sss = -5.0;
    v_sps = 4.7;
    v_pps = 5.5;
    v_ppp = -1.55;
    sum_of_phi = zeros(N, 1);
    for n1 = 1 : N
        for k = 1 : neighbor_number(n1)
            n2 = neighbor_list(n1, k);
            r12 = r(n2, :) - r(n1, :);
            r12 = r12 - round(r12./L).*L_times_pbc; % minimum image convention
            d12 = sqrt(sum(r12.*r12));
            sum_of_phi(n1) = sum_of_phi(n1) + phi(d12);
        end
    end
    for n1 = 1 : N
        H(4*n1 - 3 : 4*n1, 4*n1 - 3 : 4*n1) = on_site_matrix;
        for k = 1 : neighbor_number(n1)
            n2 = neighbor_list(n1, k);
            r12 = r(n2, :) - r(n1, :);
            r12 = r12 - round(r12./L).*L_times_pbc; % minimum image convention
            d12 = sqrt(sum(r12.*r12));
            cos_x=r12(1)/d12; cos_y=r12(2)/d12; cos_z=r12(3)/d12;
            cos_xx=cos_x*cos_x; cos_yy=cos_y*cos_y; cos_zz=cos_z*cos_z;
            sin_xx=1-cos_xx; sin_yy=1-cos_yy; sin_zz=1-cos_zz;
            cos_xy=cos_x*cos_y; cos_yz=cos_y*cos_z; cos_zx=cos_z*cos_x;
            H12(1, 1) = v_sss;
            H12(2, 2) = v_pps * cos_xx + v_ppp * sin_xx;
            H12(3, 3) = v_pps * cos_yy + v_ppp * sin_yy;
            H12(4, 4) = v_pps * cos_zz + v_ppp * sin_zz;
            H12(1, 2) = v_sps * cos_x;
            H12(1, 3) = v_sps * cos_y;
            H12(1, 4) = v_sps * cos_z;
            H12(2, 3) = (v_pps - v_ppp) * cos_xy;
            H12(3, 4) = (v_pps - v_ppp) * cos_yz;
            H12(4, 2) = (v_pps - v_ppp) * cos_zx;
            H12(2, 1) = - H12(1, 2);
            H12(3, 1) = - H12(1, 3);
            H12(4, 1) = - H12(1, 4);
            H12(3, 2) = H12(2, 3);
            H12(4, 3) = H12(3, 4);
            H12(2, 4) = H12(4, 2);
            H0(n1*4-3 : n1*4, n2*4-3 : n2*4) = H12;
            H(n1*4-3 : n1*4, n2*4-3 : n2*4) = s(d12) * H12;
            temp = phi_d(d12)*(f_d(sum_of_phi(n1))+f_d(sum_of_phi(n2)))/d12;
            force(n1, :) = force(n1, :) + r12 * temp;
        end
        energy = energy + f(sum_of_phi(n1));
    end
    [C, E] = eig(H);
    E_diag=diag(E);
    energy = energy + 2*sum(E_diag(1:N2)); % 2 = spin degeneracy
    for n1 = 1 : N
        for k = 1 : neighbor_number(n1)
            n2 = neighbor_list(n1, k);
            r12 = r(n2, :) - r(n1, :);
            r12 = r12 - round(r12./L).*L_times_pbc; % minimum image convention
            d12 = sqrt(sum(r12.*r12));
            cos_x=r12(1)/d12; cos_y=r12(2)/d12; cos_z=r12(3)/d12;
            cos_xx=cos_x*cos_x; cos_yy=cos_y*cos_y; cos_zz=cos_z*cos_z;
            sin_xx=1-cos_xx; sin_yy=1-cos_yy; sin_zz=1-cos_zz;
            cos_xy=cos_x*cos_y; cos_yz=cos_y*cos_z; cos_zx=cos_z*cos_x;
            cos_xyz = cos_xy * cos_z;
            e_sx = [sin_xx, -cos_xy, -cos_zx];
            e_sy = [-cos_xy, sin_yy, -cos_yz];
            e_sz = [-cos_zx, -cos_yz, sin_zz];
            e_xx = 2*cos_x*e_sx; e_yy = 2*cos_y*e_sy; e_zz=2*cos_z*e_sz;
            e_xy = [cos_y*(1-2*cos_xx), cos_x*(1-2*cos_yy), -2*cos_xyz];
            e_yz = [-2*cos_xyz, cos_z*(1-2*cos_yy), cos_y*(1-2*cos_zz)];
            e_zx = [cos_z*(1-2*cos_xx), -2*cos_xyz, cos_x*(1-2*cos_zz)];
            F = zeros(4, 4);
            for a = 1 : 4
                for b = 1 : 4
                    for n = 1 : N2
                        F(a, b)=F(a, b)+C(4*(n1-1)+a, n)*C(4*(n2-1)+b, n);
                    end
                end
            end
            K1 = zeros(4, 4, D);
            K  = zeros(4, 4, D);
            for d = 1 : D
                K1(:, :, d) = H0(4*(n1-1)+1:4*n1, 4*(n2-1)+1:4*n2);
                K1(:, :, d) = K1(:, :, d) * (s_d(d12) * r12(d) / d12);
                K(2, 2, d) = s(d12)/d12*(v_pps - v_ppp)*e_xx(d);
                K(3, 3, d) = s(d12)/d12*(v_pps - v_ppp)*e_yy(d);
                K(4, 4, d) = s(d12)/d12*(v_pps - v_ppp)*e_zz(d);
                K(1, 2, d) = s(d12)/d12 * v_sps * e_sx(d);
                K(1, 3, d) = s(d12)/d12 * v_sps * e_sy(d);
                K(1, 4, d) = s(d12)/d12 * v_sps * e_sz(d);
                K(2, 3, d) = s(d12)/d12 * (v_pps - v_ppp) * e_xy(d);
                K(3, 4, d) = s(d12)/d12 * (v_pps - v_ppp) * e_yz(d);
                K(4, 2, d) = s(d12)/d12 * (v_pps - v_ppp) * e_zx(d);
                K(2, 1, d) = - K(1, 2, d);
                K(3, 1, d) = - K(1, 3, d);
                K(4, 1, d) = - K(1, 4, d);
                K(3, 2, d) = + K(2, 3, d);
                K(4, 3, d) = + K(3, 4, d);
                K(2, 4, d) = + K(4, 2, d);
            end
            K = K + K1;
            for d = 1 : D
                force(n1, d) = force(n1, d) + 4 * sum(sum(F .* K(:, :, d)));
            end
        end
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
function y = s(r)
    n=2.0;
    nc=6.5;
    rc=2.18;
    r0=1.536329;
    y=(r0/r)^2*exp(n*(-(r/rc)^nc+(r0/rc)^nc));
end
function y = s_d(r)
    n=2.0;
    nc=6.5;
    rc=2.18;
    y=-n*s(r)*(1+nc*(r/rc)^nc)/r;
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

