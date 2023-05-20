function q=get_q(C,x,y,z,order)
if order==4
    r=sqrt(x.^2+y.^2+z.^2);
    s3=sum(3*z.^2-r.^2);
    s4=sum(x.*z);
    s5=sum(y.*z);
    s6=sum(x.^2-y.^2);
    s7=sum(2*x.*y);
    q=C(1)*s3*s3*s3+C(2)*s3*(s4*s4+s5*s5)+C(3)*s3*(s6*s6+s7*s7)+C(4)*s6*(s5*s5-s4*s4)+C(5)*s4*s5*s7;
elseif order==5
    s0=sum(z);
    s1=sum(x);
    s2=sum(y);
    q=C(1)*s0^4+C(2)*s0^2*(-s1^2-s2^2)+C(3)*(s1^2+s2^2)^2;
end
end
