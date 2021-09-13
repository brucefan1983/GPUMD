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


xia_4ph=[
300.08830957805685, 0.38360352763989636
400.12650347056643, 0.5141101618514272
500.164697363076, 0.642545262186585
600.2028912555855, 0.7668372947689954
700.2410851480952, 0.8787001240931651];

xia_3ph=[
300.08830957805685, 0.23859615629375064
400.12650347056643, 0.292456037079462
500.164697363076, 0.33595824848330547
600.2028912555855, 0.37531739213440196
700.2410851480952, 0.4084619341563782];

figure;
errorbar(temp_nep,kappa_ave,kappa_err,'d','linewidth',2);hold on;
plot(300,1.85,'s','linewidth',2);
plot(xia_3ph(:,1),1./xia_3ph(:,2),'o','linewidth',2);
plot(xia_4ph(:,1),1./xia_4ph(:,2),'^','linewidth',2);
plot(kappa_PbTe(:,1),kappa_PbTe(:,2),'+','linewidth',2);
plot(temp_exp2,kappa_exp2,'x','linewidth',2);
legend('NEP-HNEMD','Zeng-GAP-BTE (3ph+4ph)','Xia-DFT-BTE (3ph)',...
    'Xia-DFT-BTE (3ph+4ph)','Fedorov-Experiments','El-Sharkawy-Experiments')
xlabel('$T$ (K)','interpreter','latex');
ylabel('$\kappa$ (W/mK)','interpreter','latex');
ylim([0, 6]);
xlim([290,710]);
set(gca,'fontsize',15);





