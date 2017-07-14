clear; close all; font_size=10;

load hac.out;
N=5000; % number of correlation steps
M=length(hac)/N % number of independent functions

t=hac(1:N,1); % correlation time

hac_i = ( mean(reshape(hac(:,2),N,M),2) + mean(reshape(hac(:,5),N,M),2) ) / 2;
hac_o = ( mean(reshape(hac(:,3),N,M),2) + mean(reshape(hac(:,6),N,M),2) ) / 2;

rtc_i = ( mean(reshape(hac(:,9),N,M),2) + mean(reshape(hac(:,12),N,M),2) ) / 2;
rtc_o = ( mean(reshape(hac(:,10),N,M),2) + mean(reshape(hac(:,13),N,M),2) ) / 2;
rtc_c = ( mean(reshape(hac(:,11),N,M),2) + mean(reshape(hac(:,14),N,M),2) ) / 2;
rtc_z = mean(reshape(hac(:,15),N,M),2);

figure
% decompose
subplot(1,2,1);
loglog(t, hac_i/hac_i(1), 'r-', t, hac_o/hac_o(1), 'b--', 'linewidth',1);
xlabel('Correlation time (ps)','fontsize',font_size);
ylabel('Normalized HAC','fontsize',font_size);
xlim([0.1,1100]);
ylim([10^(-4),1]);
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);
legend('in','out');
title('(a)');

subplot(1,2,2);
semilogx(t, rtc_i, 'r-', t, rtc_o, 'b--', t, rtc_c, 'g-.', t, rtc_z, 'k:', 'linewidth', 1);
xlabel('Correlation Time (ps)','fontsize',font_size);
ylabel('\kappa (W/mK)','fontsize',font_size);
xlim([0,1100]);
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);
legend('in','out','cross','z');
title('(b)');



