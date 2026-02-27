

close all;
clear all;


%% Define the time signal

%Define the time variable
dt = 0.001;
t = 0:dt:1;

%Define signal
f1 = 50;
f2 = 120;
S = 2*cos(2*pi*f1*t) + sin(2*pi*f2*t);

%add noise to the clean signal
Snoise = S + 2.5*randn(size(t));

%visulaise both the clean and the noisy signals
figure;
subplot(3,1,1)
plot(t,S,'k','LineWidth',1.5)
xlabel('Time','interpreter','latex')
ylabel('$S(t)$','interpreter','latex')
xlim([0 , 0.3])
ylim([-5 , 5])
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
hold on
plot(t,Snoise,'r','LineWidth',1)
legend('Clean','Noisy')


%% Compute the Fast Fourier Transform FFT of the signal

%Define the number of points in the frequency domain
n = length(t);

%Fourier Transform of the noisy signal
St = fft(Snoise,n);

%create the x-axis of the frequencies in Hz (wave number)
k = 1/(dt*n)*(0:n-1);


%% Use PSD to filter out noise

%calculate the Power Spectrum Density (Power per Frequency)
%PSD is the normalised squared magnitude of S
PSD = (St.*conj(St))/n;

%find all frequecnies with large power
indices = PSD > 100;

%zero out all others
PSDFiltered = PSD.*indices;

%visualise the PSD
figure(1)
hold on
subplot(3,1,2)
plot(k,PSD,'r','LineWidth',1.5)
xlabel('Frequency','interpreter','latex')
ylabel('PSD','interpreter','latex')
xlim([0 , 500])
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
hold on
plot(k, PSDFiltered,'k','LineWidth',1.5)
legend('Noisy','Filtered')


%% Go back to the time signal and visualise the results of PSD filtering

%filter the time signal based on the pre-defined indicies
St = indices.*St;

%inverse FFT for filtered time signal
Sti = ifft(St);

%visualise the comparison between the clean and the filtered signal
figure(1)
hold on
subplot(3,1,3)
plot(t,S,'k','LineWidth',1.5)
xlabel('Time','interpreter','latex')
ylabel('$S(t)$','interpreter','latex')
xlim([0 , 0.3])
ylim([-5 , 5])
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
hold on
plot(t, Sti,'b-','LineWidth',0.7)
legend('Clean','Filtered')

