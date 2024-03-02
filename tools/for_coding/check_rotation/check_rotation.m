clear;close all;
order=5; % can be 4 or 5
C=get_C(order);
N=10;
xyz=rand(3,N);
q_ref=get_q(C,xyz(1,:),xyz(2,:),xyz(3,:),order);
xyz_rotated=zeros(3,N);
q_rotated=zeros(1000,1);
count=0;
for alpha=linspace(0,2*pi,10)
    Rx=[1,0,0;0,cos(alpha),-sin(alpha);0,sin(alpha),cos(alpha)];
    for beta=linspace(0,2*pi,10)
        Ry=[cos(beta),0,sin(beta);0,1,0;-sin(beta),0,cos(beta)];
        for gamma=linspace(0,2*pi,10)
            Rz=[cos(gamma),-sin(gamma),0;sin(gamma),cos(gamma),0;0,0,1];
            for n=1:N
                xyz_rotated(:,n)=Rz*Ry*Rx*xyz(:,n);
            end
            count=count+1;
            q_rotated(count)=get_q(C,xyz_rotated(1,:),xyz_rotated(2,:),xyz_rotated(3,:),order);
        end
    end
end

% should be a constant 1
figure;
plot(q_rotated/q_ref)
