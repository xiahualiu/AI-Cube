clear 
close all
clc

load loss

len=length(loss);

loss=reshape(loss(1:49*20),49,[]);

figure
for i=1:49
    plot(loss(i,:))
    hold on
end
