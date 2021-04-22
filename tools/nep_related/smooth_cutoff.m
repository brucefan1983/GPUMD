clear; close all;
x=0:0.01:1;
f_poly1=1+2*x.^3-3.*x.^2;
f_poly1_derivative=6*x.^2-6*x;
f_poly2=f_poly1.^2;
f_poly2_derivative=2*f_poly1.*f_poly1_derivative;
f_tersoff=0.5+0.5*cos(pi*x);
f_tersoff_derivative=-0.5*pi*sin(pi*x);

figure;
subplot(1,2,1);
plot(x,f_poly1,'r-','linewidth',1.5);hold on;
plot(x,f_poly2,'b--','linewidth',1.5);
plot(x,f_tersoff,'g-.','linewidth',1.5);
legend('poly-1','poly-2','Tersoff');
title('Cutoff Function');

subplot(1,2,2);
plot(x,f_poly1_derivative,'r-','linewidth',1.5);hold on;
plot(x,f_poly2_derivative,'b--','linewidth',1.5);
plot(x,f_tersoff_derivative,'g-.','linewidth',1.5);
legend('poly-1','poly-2','Tersoff');
title('Derivative of the Cutoff Function');




