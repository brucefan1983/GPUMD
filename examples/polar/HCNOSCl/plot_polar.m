clear;close all;
% alphaML results:
load alphaML/diag.txt; load alphaML/test_diag.txt; load alphaML/offdiag.txt; load alphaML/test_offdiag.txt;
% NEP results:
load polarizability_train.out; load polarizability_test.out; 

N=size(polarizability_train,1)/2;
train_nep=reshape(polarizability_train(:,1),N,2);
train_dft=reshape(polarizability_train(:,2),N,2);
N=size(polarizability_test,1)/2;
test_nep=reshape(polarizability_test(:,1),N,2);
test_dft=reshape(polarizability_test(:,2),N,2);

figure;

subplot(1,2,1);
plot([diag(:,1);offdiag(:,1)],[diag(:,2);offdiag(:,2)],'o','markersize',6);hold on;
plot(polarizability_train(:,2),polarizability_train(:,1),'.','markersize',10); hold on;
plot(-10:0.01:30,-10:0.01:30,'linewidth',2);
xlabel('Target polarizability/atom','fontsize',15,'interpreter','latex');
ylabel('Predicted polarizability/atom','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
legend('alphaML','NEP');
axis tight;

subplot(1,2,2);
plot([test_diag(:,1);test_offdiag(:,1)],[test_diag(:,2);test_offdiag(:,2)],'o','markersize',6);hold on;
plot(polarizability_test(:,2),polarizability_test(:,1),'.','markersize',10); hold on;
plot(-10:0.01:20,-10:0.01:20,'linewidth',2);
xlabel('Target polarizability/atom','fontsize',15,'interpreter','latex');
ylabel('Predicted polarizability/atom','fontsize',15,'interpreter','latex');
set(gca,'fontsize',15,'ticklength',get(gca,'ticklength')*2);
legend('alphaML','NEP');
axis tight;

disp(['MAE_alphaml_train_diag=',num2str(mean(abs(diag(:,1)-diag(:,2))))])
disp(['MAE_alphaml_train_offdiag=',num2str(mean(abs(offdiag(:,1)-offdiag(:,2))))])
disp(['MAE_NEP_train_diag=',num2str(mean(abs(train_dft(:,1)-train_nep(:,1))))])
disp(['MAE_NEP_train_offdiag=',num2str(mean(abs(train_dft(:,2)-train_nep(:,2))))])

disp(['MAE_alphaml_test_diag=',num2str(mean(abs(test_diag(:,1)-test_diag(:,2))))])
disp(['MAE_alphaml_test_offdiag=',num2str(mean(abs(test_offdiag(:,1)-test_offdiag(:,2))))])
disp(['MAE_NEP_test_diag=',num2str(mean(abs(test_dft(:,1)-test_nep(:,1))))])
disp(['MAE_NEP_test_offdiag=',num2str(mean(abs(test_dft(:,2)-test_nep(:,2))))])
