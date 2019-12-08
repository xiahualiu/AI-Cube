clear 
close all
clc

load loss

len=length(loss);
d1=loss(1:8:len);
d2=loss(2:8:len);
d3=loss(3:8:len);
d4=loss(4:8:len);
d5=loss(5:8:len);
d6=loss(6:8:len);
d7=loss(7:8:len);
d8=loss(8:8:len);

sum=sum([d1;d2;d3;d4;d5;d6;d7;d8]);

figure
plot(sum)
