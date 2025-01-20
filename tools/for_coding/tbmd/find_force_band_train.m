function [energy, force] = find_force_band_train(N, neighbor_number, neighbor_list, box, rc, r, para)
D=3;
N_des=5;
N4 = N*4; N2 = N4/2; H0 = zeros(N4, N4); H  = zeros(N4, N4);
energy = 0; force = zeros(N, D);

[sum_g] = find_sum_g(N, neighbor_number, neighbor_list, box, rc, r, N_des);
onsite=zeros(2,N);
onsite_p=zeros(2,N_des,N);
for n1 = 1 : N
    [onsite(:,n1), onsite_p(:,:,n1)] = onsite_ann(sum_g(n1,:), para(1:50+(N_des-1)*5), N_des);
end

for n1 = 1 : N
    H(4*n1 - 3 : 4*n1, 4*n1 - 3 : 4*n1) = diag([onsite(1,n1), onsite(2,n1), onsite(2,n1), onsite(2,n1)]);
    for k = 1 : neighbor_number(n1)
        n2 = neighbor_list(n1, k);
        r12 = r(n2, :) - r(n1, :);
        r12 = r12 - round(r12./box).*box; % minimum image convention
        d12 = sqrt(sum(r12.*r12));
        cos_x=r12(1)/d12; cos_y=r12(2)/d12; cos_z=r12(3)/d12;
        cos_xx=cos_x*cos_x; cos_yy=cos_y*cos_y; cos_zz=cos_z*cos_z;
        sin_xx=1-cos_xx; sin_yy=1-cos_yy; sin_zz=1-cos_zz;
        cos_xy=cos_x*cos_y; cos_yz=cos_y*cos_z; cos_zx=cos_z*cos_x;

        [s,sd]=hopping_scaling(rc, d12,para(50+(N_des-1)*5+1:end));

        v_sss = s(1);
        v_sps = s(2);
        v_pps = s(3);
        v_ppp = s(4);

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
        H(n1*4-3 : n1*4, n2*4-3 : n2*4) = H12;

        % redefine
        v_sss = sd(1);
        v_sps = sd(2);
        v_pps = sd(3);
        v_ppp = sd(4);

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
    end
end
[C, E] = eig(H);
E_diag=diag(E);
energy = energy + 2*sum(E_diag(1:N2)); % 2 = spin degeneracy
for n1 = 1 : N
    F_onsite_n1 = zeros(4, 4);
    for a = 1 : 4
        for b = 1 : 4
            for n = 1 : N2
                F_onsite_n1(a, b)=F_onsite_n1(a, b)+C(4*(n1-1)+a, n)*C(4*(n1-1)+b, n);
            end
        end
    end

    for k = 1 : neighbor_number(n1)
        n2 = neighbor_list(n1, k);
        r12 = r(n2, :) - r(n1, :);
        r12 = r12 - round(r12./box).*box; % minimum image convention
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

        [s]=hopping_scaling(rc, d12,para(50+(N_des-1)*5+1:end));
        v_sps = s(2);
        v_pps = s(3);
        v_ppp = s(4);

        for d = 1 : D
            K1(:, :, d) = H0(4*(n1-1)+1:4*n1, 4*(n2-1)+1:4*n2);
            K1(:, :, d) = K1(:, :, d) * (r12(d) / d12);
            K(2, 2, d) = 1/d12*(v_pps - v_ppp)*e_xx(d);
            K(3, 3, d) = 1/d12*(v_pps - v_ppp)*e_yy(d);
            K(4, 4, d) = 1/d12*(v_pps - v_ppp)*e_zz(d);
            K(1, 2, d) = 1/d12 * v_sps * e_sx(d);
            K(1, 3, d) = 1/d12 * v_sps * e_sy(d);
            K(1, 4, d) = 1/d12 * v_sps * e_sz(d);
            K(2, 3, d) = 1/d12 * (v_pps - v_ppp) * e_xy(d);
            K(3, 4, d) = 1/d12 * (v_pps - v_ppp) * e_yz(d);
            K(4, 2, d) = 1/d12 * (v_pps - v_ppp) * e_zx(d);
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

        F_onsite_n2 = zeros(4, 4);
        for a = 1 : 4
            for b = 1 : 4
                for n = 1 : N2
                    F_onsite_n2(a, b)=F_onsite_n2(a, b)+C(4*(n2-1)+a, n)*C(4*(n2-1)+b, n);
                end
            end
        end

        for d = 1 : D
            for m = 1 : N_des
                onsite_p_n1_diag = diag([onsite_p(1,m,n1), onsite_p(2,m,n1), onsite_p(2,m,n1), onsite_p(2,m,n1)]);
                onsite_p_n2_diag = diag([onsite_p(1,m,n2), onsite_p(2,m,n2), onsite_p(2,m,n2), onsite_p(2,m,n2)]);
                dgdr = (m*2) * (d12 / rc - 1)^(m*2-1) / rc / d12;
                force(n1, d) = force(n1, d) + 2 * sum(sum(F_onsite_n1 .* onsite_p_n1_diag + F_onsite_n2 .* onsite_p_n2_diag)) * dgdr * r12(d);
            end
        end
    end
end
end

function [y, yp] = hopping_scaling(rc, r, para)
q = (r / rc - 1)^2;
N_neurons = 5;
w0 = para(1 : N_neurons);
offset = N_neurons;
b0 = para(offset+1 : offset + N_neurons);
offset = offset + N_neurons;
w1 = reshape(para(offset+1 : offset + N_neurons*N_neurons), N_neurons, N_neurons);
offset = offset + N_neurons*N_neurons;
b1 = para(offset+1 : offset + N_neurons);
offset = offset + N_neurons;
w2 = reshape(para(offset+1 : offset + N_neurons*4), 4, N_neurons);
y = zeros(4, 1);
yp = zeros(4, 1);
x1 = zeros(1, N_neurons);
x2 = zeros(1, N_neurons);
for n0 = 1 : N_neurons
    x1(n0) = tanh(w0(n0) * q - b0(n0));
end
for n1 = 1 : N_neurons
    w1_times_x1 = 0.0;
    for n0 = 1 : N_neurons
        w1_times_x1 = w1_times_x1 + w1(n1, n0) * x1(n0);
    end
    x2(n1) = tanh(w1_times_x1 - b1(n1));
    for k = 1 : 4
        y(k) = y(k) + w2(k, n1) * x2(n1);
    end
end

dydx1 = zeros(4, N_neurons);
for n0 = 1 : N_neurons
    for k = 1 : 4
        temp_sum = 0.0;
        for n1 = 1 : N_neurons
            temp_sum = temp_sum + w2(k, n1) * (1 - x2(n1) * x2(n1)) * w1(n1, n0);
        end
        dydx1(k, n0) = temp_sum;
    end
end

for k = 1 : 4
    temp_sum = 0.0;
    for n0 = 1 : N_neurons
        temp_sum = temp_sum + dydx1(k, n0) * (1.0 - x1(n0) * x1(n0)) * w0(n0);
    end
    yp(k) = temp_sum * 2 * (r / rc - 1) / rc;
end

end


function [sum_g] = find_sum_g(N, neighbor_number, neighbor_list, box, rc, r, N_des)
sum_g = zeros(N,N_des);
for n1 = 1 : N
    for k = 1 : neighbor_number(n1)
        n2 = neighbor_list(n1, k);
        r12 = r(n2, :) - r(n1, :);
        r12 = r12 - round(r12./box).*box; % minimum image convention
        d12 = sqrt(sum(r12.*r12));
        for d = 1 : N_des
            sum_g(n1,d) = sum_g(n1,d) + (d12/rc-1).^(2*d);
        end
    end
end
end

function [y, yp] = onsite_ann(q, para, N_des)
N_neurons = 5;
w0 = reshape(para(1 : N_neurons*N_des),N_neurons,N_des);
offset = N_neurons*N_des;
b0 = para(offset+1 : offset + N_neurons);
offset = offset + N_neurons;
w1 = reshape(para(offset+1 : offset + N_neurons*N_neurons), N_neurons, N_neurons);
offset = offset + N_neurons*N_neurons;
b1 = para(offset+1 : offset + N_neurons);
offset = offset + N_neurons;
w2 = reshape(para(offset+1 : offset + N_neurons*2), 2, N_neurons);
y = zeros(2, 1);
yp = zeros(2, N_des);
x1 = zeros(1, N_neurons);
x2 = zeros(1, N_neurons);
for n0 = 1 : N_neurons
    w0_times_q = 0.0;
    for d = 1 : N_des
      w0_times_q = w0_times_q + w0(n0, d) * q(d);
    end
    x1(n0) = tanh(w0_times_q - b0(n0));
end
for n1 = 1 : N_neurons
    w1_times_x1 = 0.0;
    for n0 = 1 : N_neurons
        w1_times_x1 = w1_times_x1 + w1(n1, n0) * x1(n0);
    end
    x2(n1) = tanh(w1_times_x1 - b1(n1));
    for k = 1 : 2
        y(k) = y(k) + w2(k, n1) * x2(n1);
    end
end

dydx1 = zeros(4, N_neurons);
for n0 = 1 : N_neurons
    for k = 1 : 2
        temp_sum = 0.0;
        for n1 = 1 : N_neurons
            temp_sum = temp_sum + w2(k, n1) * (1 - x2(n1) * x2(n1)) * w1(n1, n0);
        end
        dydx1(k, n0) = temp_sum;
    end
end

for k = 1 : 2
    for d = 1 : N_des
        temp_sum = 0.0;
        for n0 = 1 : N_neurons
            temp_sum = temp_sum + dydx1(k, n0) * (1.0 - x1(n0) * x1(n0)) * w0(n0, d);
        end
        yp(k,d) = temp_sum;
    end
end

end

