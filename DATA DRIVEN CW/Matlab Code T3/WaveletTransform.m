

close all;
clear all;



%% Define the domain variables

%the range of the time domain
L = 10; 

%number of points in the time domain
n = 2048; 


%% Apply Continous Wavelet: Mexican Hat Wavelet

%Define the time variable
t2c = linspace(0,L,n+1);
tc = t2c(1:n);

%Define signal
Sc = (3*sin(2*tc)+0.5*tanh(0.5*(tc-3))+0.2*exp(-(tc-4).^2)+1.5*sin(5*tc)+4*cos(3*(tc-6).^2))/10+(tc/20).^3;

% Define scales manually for Mexican Hat wavelet
scales = 1:127;

% Compute CWT using Mexican Hat wavelet manually
coefficients1c = cwt(Sc, scales, 'mexh');


%% Apply Discrete Wavelet: Haar Wavelet

%Define the time variable
t2d = linspace(0,L,2*n+1);
td = t2d(1:2*n);

%Define signal
Sd = (3*sin(2*td)+0.5*tanh(0.5*(td-3))+0.2*exp(-(td-4).^2)+1.5*sin(5*td)+4*cos(3*(td-6).^2))/10+(td/20).^3;

% Apply Discrete Wavelet Transform (DWT) using Haar wavelet
[coefficients1d, ~] = dwt(Sd, 'sym4');


%% plotting
figure;
% Plot Mexican hat wavelet transform
subplot(3,1,1);
imagesc(abs(coefficients1c) / max(abs(coefficients1c(:))));
colormap jet;
axis tight;
ylabel('Scale');
title('Mexican Continuous Wavelet');
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
% Plot Haar wavelet transform
subplot(3,1,2);
imagesc(abs(coefficients1d) / max(abs(coefficients1d(:))));
colormap jet;
axis tight;
ylabel('Scale');
title('Haar Discrete Wavelet');
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
% Plot the original signal
subplot(3,1,3);
plot(Sc,'k','LineWidth',1.5)
title('Signal in time domain')
xlabel('Time','interpreter','latex') 
xlim([0 length(tc)])
ylabel('$S(t)$','interpreter','latex')
title('Original Signal');
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
grid on

