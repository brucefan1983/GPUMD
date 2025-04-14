clear; close all;

data = table2array(readtable('thermo.out', 'FileType', 'text', 'CommentStyle', '#'));

t = (1:size(data,1))*0.01;

figure;
plot(t, data(:,1));
xlabel('Time (ps)');
ylabel('Temperature (K)');

