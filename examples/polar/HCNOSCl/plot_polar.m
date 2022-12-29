clear;close all;
% alphaML results:
load alphaML/diag.txt; load alphaML/test_diag.txt; load alphaML/offdiag.txt; load alphaML/test_offdiag.txt;
% NEP results:
load loss.out; load polarizability_train.out; load polarizability_test.out; 

N=size(polarizability_train,1)/2;
train_nep=reshape(polarizability_train(:,1),N,2);
train_dft=reshape(polarizability_train(:,2),N,2);
N=size(polarizability_test,1)/2;
test_nep=reshape(polarizability_test(:,1),N,2);
test_dft=reshape(polarizability_test(:,2),N,2);

figure;
subplot(2,2,1);
loglog(sum(loss(:,3:4),2),'-','linewidth',4);hold on;
loglog(loss(:,7),'-','linewidth',4);hold on;
loglog(loss(:,10),'-','linewidth',4);hold on;
xlabel('Generation/100','interpreter','latex');
ylabel('Loss','interpreter','latex');
set(gca,'fontsize',15,'linewidth',1.5);
legend('Regularization','Polarizibility-Train','Polarizibility-Test');
axis tight

subplot(2,2,2);
plot([diag(:,1);offdiag(:,1)],[diag(:,2);offdiag(:,2)],'o','markersize',6);hold on;
plot(polarizability_train(:,2),polarizability_train(:,1),'.','markersize',10); hold on;
plot(-10:0.01:30,-10:0.01:30,'linewidth',2);
xlabel('Target polarizability/atom','fontsize',15,'interpreter','latex');
ylabel('Predicted polarizability/atom','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
legend('alphaML','NEP');
axis tight;

subplot(2,2,3);
plot([test_diag(:,1);test_offdiag(:,1)],[test_diag(:,2);test_offdiag(:,2)],'o','markersize',6);hold on;
plot(polarizability_test(:,2),polarizability_test(:,1),'.','markersize',10); hold on;
plot(-10:0.01:20,-10:0.01:20,'linewidth',2);
xlabel('Target polarizability/atom','fontsize',15,'interpreter','latex');
ylabel('Predicted polarizability/atom','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
legend('alphaML','NEP');
axis tight;


disp(['MAE_alphaml_diag=',num2str(mean(abs(test_diag(:,1)-test_diag(:,2))))])
disp(['MAE_alphaml_offdiag=',num2str(mean(abs(test_offdiag(:,1)-test_offdiag(:,2))))])
disp(['MAE_NEP_diag=',num2str(mean(abs(test_dft(:,1)-test_nep(:,1))))])
disp(['MAE_NEP_offdiag=',num2str(mean(abs(test_dft(:,2)-test_nep(:,2))))])

