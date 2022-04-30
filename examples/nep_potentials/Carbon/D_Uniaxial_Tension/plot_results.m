clear; close all; font_size = 15;

fid = fopen('Uniaxial_Tension_Results.txt');
data = cell2mat( textscan(fid, '%f%f', 'CommentStyle', '#', 'CollectOutput', true) );
fclose(fid);

figure;
plot(data(:,1),data(:,2),'-','linewidth',2); hold on;
xlabel('Strain','fontsize',font_size,'interpreter','latex');
ylabel('Stress (GPa)','fontsize',font_size,'interpreter','latex');
set(gca,'fontsize',font_size,'linewidth',1.5);
xlim([0,0.35]);
ylim([0,220]);
