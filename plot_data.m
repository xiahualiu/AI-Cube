clear 
close all
clc

data=[1,1,1,1,1,1,1,1,1,0.95,0.80,0.9,0.85,13/20,12/20,14/20,17/20,16/20,14/20,13/20,13/20];

figure
plot(data.*100)
xlabel('The scramble number')
ytickformat('percentage')
ylim([0,105])
grid on
title('Difficulty vs Solve Percentage')