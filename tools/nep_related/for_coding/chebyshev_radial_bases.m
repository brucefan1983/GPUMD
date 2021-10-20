clear; close all;
n_max = 9;
r = 0 : 0.001 : 1;
x = 2 * (r-1).^2 - 1;
fc=0.5*cos(pi*r)+0.5;

f = zeros(n_max, length(x));
f(1,:) = 1;
f(2,:) = x;
for n = 3 : n_max
    f(n,:) = 2 * x .* f(n-1,:) - f(n-2,:);
end
for n = 1 : n_max
    f(n,:) = (1+f(n,:))/2 .* fc;
end

figure;
plot(r,x,'-','linewidth',1.5);
xlabel('r/r_{cut}');
ylabel('x');
set(gca,'fontsize',15);

figure;
plot(r,f,'-','linewidth',1.5);hold on;
xlabel('r/r_{cut}');
ylabel('f_n(r)');
set(gca,'fontsize',15);
