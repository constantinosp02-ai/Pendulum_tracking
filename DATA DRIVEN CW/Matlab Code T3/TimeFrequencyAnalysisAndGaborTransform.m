close all;
clear all;



%% Define the time signal

%Time domain
L = 10; %is the range of the time domain

%number of points in the time domain
n = 2048; 

%Define the time variable
t2 = linspace(0,L,n+1);
t = t2(1:n);

%Define signal
%optione1: arbitirary function/signal
S = (3*sin(2*t)+0.5*tanh(0.5*(t-3))+0.2*exp(-(t-4).^2)+1.5*sin(5*t)+4*cos(3*(t-6).^2))/10+(t/20).^3;

%option2: Quadratic Chirp signal
% f0 = 50;
% f1 = 250;
% S = cos(2*pi*t.*(f0+(f1-f0)*t.^2/(3*L^2)));

%plotting the signal
figure(1)
subplot(2,1,1)
plot(t,S,'k','LineWidth',1.5)
title('Signal in time domain')
xlabel('Time','interpreter','latex')
ylabel('$S(t)$','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
hold on

%% Compute the Frequencies of the signal 

%Define wave number
% % Option 1: frequency's unit here is rad. sec^{-1}
% k = (2*pi./L).*[0:n/2-1 -n/2:-1]; k = fftshift(k); 

% Option 2: frequency's unit here is sec^{-1}
k = (1/L)*[0:n/2-1 -n/2:-1];  k = fftshift(k); 

% % Option 3: frequency's unit here is sec^{-1} and starts from 0
% dt = 0.001; k = 1/(dt*n)*(0:n-1);

%Fourier Transform of the signal
St = fft(S);

%plotting the signa in its frequency domain
subplot(2,1,2)
plot(k,abs(fftshift(St))/max(abs(fftshift(St))),'b','LineWidth',1.5); %normalised
title('Signal in frequency domain')
xlabel('Frequency (Hz)','interpreter','latex')
ylabel('$|F(\omega)|$','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)


%% Time frequency analysis (spectrogram)

% %A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time. 
% %It demonstrates the intensity of a given signal and where in the signal occurs. In other words, it shows in the same plot the 
% %frequency content along with their corresponding temporal information.
% %One known application to this is Shazam. Shazam finds the peaks in the power spectrum of a given song and tries to match this kind
% %of sparse templets of peaks to a library of known songs.
% %Other applications: difference between male and female voices, difference between acoustic and electric guitars, difference between the sound
% %of hot and cold water pouring,..etc.

%Initialise spectrogram matrix
Sgt_spec = []; 

%Define the time variable for iteration
tslide = 0:0.1:L; 

%Loop over all time steps
for j=1:length(tslide)

%Gabor Transform/window function
width = 1; %the width of the window
g = exp(-(t-tslide(j)).^2/width^2); %Gabor function

%The broader the window/filter, we have more accurate on frequency content 
%at the sacrifice of less accurate localisation on where the signal is in
%time. So, we can trust the information along the Y domain rather than the
%informtion o the X domain. For broad window we get the low frequency component
%that narrow filter usually throws them out becasue low frequency components are 
%not locaslised in time.

%the narrower the window, we have more accurate information on the time
%domain (time localisation) and less information on the frequency, i.e., poor frequency
%resolution. In this kind of filters, a lot of frequency components are thrown away.

%Gabor Signal
Sg = g.*S; 

%Fourier Transofrm of the resulting signal
Sgt = fft(Sg);

%stor the spectrogram data at each iteration
Sgt_spec = [Sgt_spec;abs(fftshift(Sgt))/max(abs(fftshift(Sgt)))]; %matrix of spectrogram
%with rows to the time snapshots and coloums of Gabor signal

%plotting
figure(2)
%the signal and the sliding window function
subplot(3,1,1)
plot(t,S,'k',t,g,'r','LineWidth',1)
title('Signal with Gabor function in time domain')
xlabel('Time','interpreter','latex')
ylabel('$S(t)$','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
xlim([0 L])
legend('Signal','Gabor Window')
%the convolution of the signal with the window function
subplot(3,1,2)
plot(t,Sg,'m','LineWidth',1)
xlabel('Time','interpreter','latex')
ylabel('$S(t)*g(t)$','interpreter','latex')
xlim([0 L])
title('Convolution of the Signal with Gabor Transform in time domain')
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
%the normalised resulting signal in frequency domain
subplot(3,1,3)
plot(k,abs(fftshift(Sgt))/max(abs(fftshift(Sgt))),'b','LineWidth',1)
xlabel('Frequency','interpreter','latex')
ylabel('$|F(\omega)|$','interpreter','latex')
title('Fourier Transofrm of the new signal in frequency domain')
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)

%show the rsult for each iteration
drawnow

pause(0.1)
end

%plot the spectrogram along with its data
figure;
%time signal
subplot(2,1,1)
plot(t,S,'k','LineWidth',1)
xlim([0 L])
xlabel('Time','interpreter','latex')
ylabel('$S(t)$','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
%spectrogram
subplot(2,1,2)
pcolor(tslide,k,Sgt_spec')
shading interp
colormap jet
xlim([0 L])
xlabel('Time','interpreter','latex')
ylabel('Frequency','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
% ylim([-10 10])