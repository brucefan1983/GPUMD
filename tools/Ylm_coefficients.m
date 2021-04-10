clear; close all;
c=zeros(9,1);
% L = 0
c(1) = 1/2 * sqrt(1/pi);
% L = 1
c(2) = 1/2 * sqrt(3/pi);
c(3) = -1/2 * sqrt(3/pi/2);
c(4) = c(3);
% L = 2
c(5) = 1/4 * sqrt(5/pi);
c(6) = -1/2 * sqrt(15/pi/2);
c(7) = c(6);
c(8) = 1/4 * sqrt(15/pi/2);
c(9) = c(8);

format long;
disp(c);

figure;
plot(c,'o-','linewidth',2);

