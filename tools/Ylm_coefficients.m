clear; close all;
c=zeros(16,1);
% L = 0
c(1) = 1/4 * 1/pi;
% L = 1
c(2) = 1/4 * 3/pi;
c(3) = 1/4 * 3/pi/2;
c(4) = c(3);
% L = 2
c(5) = 1/16 * 5/pi;
c(6) = 1/4 * 15/pi/2;
c(7) = c(6);
c(8) = 1/16 * 15/pi/2;
c(9) = c(8);
% L = 3
c(10) = 1/16 * 7/pi;
c(11) = 1/64 * 21/pi;
c(12) = c(11);
c(13) = 1/16 * 105/pi/2;
c(14) = c(13);
c(15) = 1/64 * 35/pi;
c(16) = c(15);

format long;
disp(c);

figure;
plot(c,'o-','linewidth',2);

