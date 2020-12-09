clear
load f_dft.txt
load compute.out

f_tersoff=reshape(compute(1,:),512,3);

figure(1)
plot(f_dft(:,4:6),f_tersoff,'o'),axis([-2, 2, -2, 2])
aa=polyfit(f_dft(:,4:6),f_tersoff,1)
hold on
plot(f_dft(:,4:6),f_dft(:,4:6),'b-')
grid on;
xlabel('f-dft (eV/A)','fontsize',15);
ylabel('f-md (eV/A)','fontsize',15);
set(gca,'fontsize',15);
