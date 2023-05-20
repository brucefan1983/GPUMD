clear;format long;
    A10=1/2*sqrt(3/pi);
    A11=1/2*sqrt(3/pi/2);
    C0=7/15*A10^4;
    C1=28/15*A10^2*A11^2;
    C2=28/15*A11^4;
    C=([C0;C1;C2]);
    disp(C)
