clear 
close all
clc

load loss
load record

len=length(loss);

loss=reshape(loss(1:49*24),49,[]);

figure
for i=1:49
    plot(loss(i,:))
    hold on
end
xlabel('Epochs')
ylabel('Overall loss')
title('Loss curve during training')