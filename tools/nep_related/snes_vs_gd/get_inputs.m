function [x0,y0,N_neu,N_par]=get_inputs()
x0=exp(0:0.01:log(3));
y0=10./x0.^12-10./x0.^6;
N_neu=10;
N_par=N_neu*(N_neu+4)+1;