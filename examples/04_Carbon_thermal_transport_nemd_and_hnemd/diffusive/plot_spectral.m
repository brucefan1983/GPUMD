clear; close all; font_size = 15;

% parameters for NEMD
Nc=250; % Number of correlation steps
Nw=1000; % Number of frequency points
deltaT=20; % Temperature difference (K)
vol=(1.42*sqrt(3)*100)*(3*1.42*10)*3.35; % volume considered (A^3)
[K_NEMD,G]=get_K_and_G('../ballistic/shc.out',Nc,Nw,deltaT,vol);

% extra parameters for HNEMD
T=300; % Temperature (K) 
Fe=1.0e-5; % driving force parameter (1/A)
[K_HNEMD,kappa]=get_K_and_kappa('shc.out',Nc,Nw,T,Fe,vol);

% get the ballistic-to-diffusive results
len=10.^(1:0.1:6); % 10 nm to 1 mm
[lambda,kL]=get_lambda_and_kL(G,kappa,len);

% plot results
figure;
subplot(2,2,1);
plot(K_HNEMD(:,1),K_HNEMD(:,2),'b-','linewidth',2);
set(gca,'fontsize', font_size);
xlabel('Correlation time (ps)','fontsize', font_size);
ylabel('K (GW/m^2)','fontsize', font_size);
title('(a)');

subplot(2,2,2);
plot(kappa(:,1), kappa(:,2), 'b-','linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('\omega/2\pi (THz)','fontsize',font_size);
ylabel('\kappa(\omega) (W/m/K/THz)','fontsize',font_size);
xlim([0,53]);
set(gca,'ticklength',get(gca,'ticklength')*2);
title('(b)');

subplot(2,2,3);
plot(G(:,1),lambda,'b-','linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('\omega/2\pi (THz)','fontsize',font_size);
ylabel('\lambda(\omega) (nm)','fontsize',font_size);
ylim([0,6000]);
xlim([0,53]);
set(gca,'ticklength',get(gca,'ticklength')*3,'xtick',0:10:50);
title('(c)');

subplot(2,2,4);
semilogx(len/1000,kL,'b-','linewidth',1.5);
set(gca,'fontsize',font_size);
xlabel('L (\mum)','fontsize',font_size);
ylabel('\kappa (W/mK)','fontsize',font_size);
ylim([0,3200]);
xlim([1.0e-2,10^3]);
set(gca,'ticklength',get(gca,'ticklength')*3,'xtick',10.^(-2:1:6));
title('(d)');

