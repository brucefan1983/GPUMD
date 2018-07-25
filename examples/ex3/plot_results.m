clear; close all; font_size=10;

load hac.out;
N=5000; % number of correlation steps
M=length(hac)/N % number of independent functions
t=hac(1:N,1); % correlation time
hac_i_ave=(mean(reshape(hac(:,2),N,M),2)+mean(reshape(hac(:,4),N,M),2))/2;
hac_o_ave=(mean(reshape(hac(:,3),N,M),2)+mean(reshape(hac(:,5),N,M),2))/2;
rtc_i=(reshape(hac(:,7),N,M)+reshape(hac(:,9),N,M))/2;
rtc_o=(reshape(hac(:,8),N,M)+reshape(hac(:,10),N,M))/2;
rtc_i_ave=mean(rtc_i,2);
rtc_o_ave=mean(rtc_o,2);
rtc_t_ave=rtc_i_ave+rtc_o_ave;

figure
subplot(2,2,1);
loglog(t,hac_i_ave/hac_i_ave(1),'r-',t,hac_o_ave/hac_o_ave(1),'b--');
xlabel('Correlation Time (ps)','fontsize',font_size);
ylabel('Normalized HAC','fontsize',font_size);
xlim([0.1,1000]);
ylim([10^(-4),1]);
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);
legend('in','out');
title('(a)');

subplot(2,2,2);
plot(t,rtc_i,'-','color',0.5*[1,1,1]);
hold on;
plot(t,rtc_i_ave,'r-','linewidth',2);
xlabel('Correlation Time (ps)','fontsize',font_size);
ylabel('\kappa^{in} (W/mK)','fontsize',font_size);
xlim([0,1000]);
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);
title('(b)');

subplot(2,2,3);
plot(t,rtc_o,'-','color',0.5*[1,1,1]);
hold on;
plot(t,rtc_o_ave,'b--','linewidth',2);
xlabel('Correlation Time (ps)','fontsize',font_size);
ylabel('\kappa^{out} (W/mK)','fontsize',font_size);
xlim([0,1000]);
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);
title('(c)');

subplot(2,2,4);
plot(t,rtc_i_ave,'r-',t,rtc_o_ave,'b--',t,rtc_t_ave,'k:','linewidth',2);
xlabel('Correlation Time (ps)','fontsize',font_size);
ylabel('\kappa (W/mK)','fontsize',font_size);
xlim([0,1000]);
set(gca,'fontsize',font_size);
set(gca,'ticklength',get(gca,'ticklength')*2);
legend('in','out','tot');
title('(d)');



