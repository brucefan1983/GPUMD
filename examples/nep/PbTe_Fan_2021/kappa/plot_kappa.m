clear; close all;
load kappa_PbTe.txt;
load 300/kappa.out; kappa300=kappa(:,5); 
load 400/kappa.out; kappa400=kappa(:,5); 
load 500/kappa.out; kappa500=kappa(:,5); 
load 600/kappa.out; kappa600=kappa(:,5); 
load 700/kappa.out; kappa700=kappa(:,5); 

kappa_ave=[mean(kappa300),mean(kappa400),mean(kappa500),...
    mean(kappa600),mean(kappa700)];
kerr300=std(mean(reshape(kappa300,60,100)))/10;
kerr400=std(mean(reshape(kappa400,60,100)))/10;
kerr500=std(mean(reshape(kappa500,60,100)))/10;
kerr600=std(mean(reshape(kappa600,60,100)))/10;
kerr700=std(mean(reshape(kappa700,60,100)))/10;
kappa_err=[kerr300,kerr400,kerr500,kerr600,kerr700];

temp_nep=300:100:700;

temp_exp2=300:20:700;
kappa_exp2=[
2.141 
2.077 
2.021 
1.927 
1.897 
1.851
1.799 
1.748 
1.702 
1.690 
1.620 
1.580 
1.567 
1.487 
1.470 
1.426
1.384 
1.342 
1.299 
1.259 
1.212];

kappa_nep1_ave=[   1.9979    1.6916    1.1740    1.0335    0.8790];
kappa_nep1_err=[0.1613    0.1329    0.1331    0.1102    0.0961];

figure;
errorbar(temp_nep,kappa_nep1_ave,kappa_nep1_err,'s','linewidth',2);hold on;
errorbar(temp_nep,kappa_ave,kappa_err,'o','linewidth',2);hold on;
plot(kappa_PbTe(:,1),kappa_PbTe(:,2),'+','linewidth',2);
plot(temp_exp2,kappa_exp2,'x','linewidth',2);
legend('NEP1','NEP2','Fedorov-Experiments','El-Sharkawy-Experiments')
xlabel('$T$ (K)','interpreter','latex');
ylabel('$\kappa$ (W/mK)','interpreter','latex');
ylim([0, 2.5]);
xlim([290,710]);
set(gca,'fontsize',15);
