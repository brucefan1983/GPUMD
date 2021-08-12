function [x0,y0]=get_training_data()
x0=exp(0:0.01:log(3));
y0=10./x0.^12-10./x0.^6;