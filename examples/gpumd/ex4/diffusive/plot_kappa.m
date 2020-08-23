clear; close all;
load kappa.out;

M=size(kappa,1);
t=(1:M)*0.001; % ns
ki_ave=cumsum(kappa(:,3))./(1:M).';
ko_ave=cumsum(kappa(:,4))./(1:M).';
kxy_ave=cumsum(sum(kappa(:,1:2),2))./(1:M).';
kzy_ave=cumsum(kappa(:,5))./(1:M).';

figure;

subplot(2,2,1);
plot(t,kappa(:,3),'color',0.7*[1 1 1]);
hold on;
plot(t,ki_ave,'b--');
xlabel('time (ns)');
ylabel('\kappa_{in} (W/mK)');
ylim([-2000,4000]);
set(gca,'ticklength',get(gca,'ticklength')*2);
title('(a)');

subplot(2,2,2);
plot(t,kappa(:,4),'color',0.7*[1 1 1]);
hold on;
plot(t,ko_ave,'r--');
xlabel('time (ns)');
ylabel('\kappa_{out} (W/mK)');
ylim([0,4000]);
set(gca,'ticklength',get(gca,'ticklength')*2);
title('(b)');

subplot(2,2,3);
plot(t,ki_ave,'b--');
hold on;
plot(t,ko_ave,'r--');
plot(t,ki_ave+ko_ave,'k--');
xlabel('time (ns)');
ylabel('\kappa (W/mK)');
ylim([0,4000]);
legend('in','out','total');
set(gca,'ticklength',get(gca,'ticklength')*2);
title('(c)');

subplot(2,2,4);
plot(t,ki_ave+ko_ave,'k-');
hold on;
plot(t,kxy_ave,'-');
plot(t,kzy_ave,'-');
xlabel('time (ns)');
ylabel('\kappa (W/mK)');
ylim([-2000,4000]);
legend('yy','xy','zy');
set(gca,'ticklength',get(gca,'ticklength')*2);
title('(d)');
