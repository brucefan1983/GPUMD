function C=get_C()
A20=1/4*sqrt(5/pi);
A21=1/2*sqrt(15/pi/2);
A22=1/4*sqrt(15/pi/2);
C0=-sqrt(2/35)*A20^3;
C1=-6*sqrt(1/70)*A20*A21^2;
C2=6*sqrt(2/35)*A20*A22^2;
C3=6*sqrt(3/35)*A21^2*A22;
C4=-2*C3;
C=([C0;C1;C2;C3;C4]);
end
