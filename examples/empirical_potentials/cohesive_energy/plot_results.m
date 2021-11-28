clear; close all; font_size = 12;
load cohesive.out;
figure;
plot(cohesive(:,1)*5.43,cohesive(:,2)/8000,'o-');

