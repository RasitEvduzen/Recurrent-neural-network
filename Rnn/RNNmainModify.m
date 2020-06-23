clc,clear all,close all;
%% system configuration
ts = 0.001;
a = 20;
eta = 0.05;    % Learning Rate 0-1 
t = 0:ts:a;
l = length(t);
u = square(pi*t);
plant = [1 -0.25 0.1045 0.0902];
y(1:2) = 0;
for i=3:length(t)
    ph = [y(i-1) y(i-2) cos(2*u(i))+exp(-10*abs(u(i))) u(i-1)];
    y(i) = ph*plant'+.001*randn(1,1);
end
T = u';
Y = y';
% save data T Y
% clear T Y
%% RNN architecture
inp = 6;  % Input Layer Neurons 
N1 = 3;   % Middle Layer Neurons
N2 = 1;   % Output Layer Neurons
window_length = 3;
epoch = (length(t)-2)/window_length;
w1 = ones(N1,inp);
w2 = ones(N2,N1);
y_NN = 0.*y;
j = 2;
a1 = zeros(N1,length(t));
e = ones(window_length,1);
%% RNN configuration
for k=1:epoch
    for i=1:window_length
        j=j+1;
        Input_Vec(:,i)=[-y_NN(j-1) -y_NN(j-2) u(j) u(j-1) a1(1,j-1) 1]';
       
        n1 = w1*Input_Vec(:,i);
        
        a1(:,j)=tansig(n1);    % Hidden Layer Activation Function
        
        n2 = w2*a1(:,j);
        
        a2=tansig(n2);    % Output Layer Activation Function
        
        y_NN(j) = a2;
        
        e(i) = y(j) - y_NN(j);
        Y2(:,i) = 2*dtansig(n2,a2)*e(i); % For Output Layer
        Y1(:,i) = diag(dtansig(n1,a1(:,j)),0)*w2'*Y2(:,i); % For Hidden Layer        
    end
    rmsetr(k) = rms(e);
    w1 = w1 + eta*Y1*Input_Vec';
    w2 = w2 + eta*Y2(:,i)*a1(:,j)';
end
figure
subplot(221)
plot(t,y,'r')
hold on,grid minor,title('RNN Training')
plot(t,y_NN,'b')
legend('Actual Output','RNN');

subplot(222)
plot(rmsetr),grid minor
title(['Training RMS ',num2str(rmsetr(1,end))])
% return
%% Testing
a = 20;
ts = 0.001;
t = 0:ts:a;
l = length(t);
u = square(pi*t);
u = awgn(u,30);
plant = [1 -0.25 0.1045 0.0902];
epoch = (length(t)-2)/window_length;
y(1:2) = 0;
for i=3:length(t)
    ph = [y(i-1) y(i-2) cos(2*u(i))+exp(-10*abs(u(i))) u(i-1)];
    y(i) = ph*plant';
end

y_NN = 0.*y;
j = 2;
a1 = zeros(N1,length(t));
for k=1:epoch
    for i=1:window_length
        j = j + 1;
        Input_Vec(:,i) = [-y_NN(j-1) -y_NN(j-2) u(j) u(j-1) a1(1,j-1) 1]';
        n1 = w1*Input_Vec(:,i);
        
        a1(:,j) = tansig(n1);    
        
        n2 = w2*a1(:,j);
        
        a2 = tansig(n2);    
        
        y_NN(j) = a2;
        
        e(i) = y(j) - y_NN(j);
        rmseval(k) = rms(e);
    end
    
end

%% PLOT DATA

subplot(223)
plot(y(1,1:1000),'m','LineWidth',2)
hold on,grid minor,title('RNN Validation')
plot(y_NN(1,1:1000),'b','LineWidth',0.5)
legend('Actual Output','RNN');


subplot(224)
plot(rmseval),grid minor
title(['Validation RMS ',num2str(rmseval(1,end))])
% return
run MISO_YSA