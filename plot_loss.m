clear 
close all
clc

load loss

len=length(loss);
d1=1:8:len;
d2=2:8:len;
d3=3:8:len;
d4=4:8:len;
d5=5:8:len;
d6=6:8:len;
d7=7:8:len;
d8=8:8:len;

figure
plot(loss(d1))
hold on
plot(loss(d2))
plot(loss(d3))
plot(loss(d4))
plot(loss(d5))
plot(loss(d6))
plot(loss(d7))
plot(loss(d8))