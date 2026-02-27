
% This code applies both methods: PCA and SVD to reduce the dimeniosnality
% occuring on a harmonic oscillator experiement taken from three different
% orientations. Each expeiremnt produces a dataset that contains the
% informtion of the displacmenet on three different axes: x,y, and z.

close all;
clear all;

%% load the datasets
%first dataset from the first camera
HO1 = xlsread('Data1_Harmonic');
%check the size 
[row1,col1] = size(HO1);
%second dataset from the second camera
HO2 = xlsread('Data2_Harmonic');
%check the size 
[row2,col2] = size(HO2);
%third dataset from the third camera
HO3 = xlsread('Data3_Harmonic');
%check the size 
[row3,col3] = size(HO3);

%visulaise the actual dynamics.
figure;
% Experiment 1
subplot(3,3,1)
plot(HO1(:,2),'b','LineWidth',1.5)
xlabel('No of recordings','interpreter','latex')
ylabel('$x$ axis','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
grid on
subplot(3,3,2)
plot(HO1(:,3),'b','LineWidth',1.5)
xlabel('No of recordings','interpreter','latex')
ylabel('$y$ axis','interpreter','latex')
title('Experiement 1','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
grid on
subplot(3,3,3)
plot(HO1(:,4),'b','LineWidth',1.5)
xlabel('No of recordings','interpreter','latex')
ylabel('$z$ axis','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
grid on
% Experiment 2
subplot(3,3,4)
plot(HO2(:,2),'r','LineWidth',1.5)
xlabel('No of recordings','interpreter','latex')
ylabel('$x$ axis','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
grid on
subplot(3,3,5)
plot(HO2(:,3),'r','LineWidth',1.5)
xlabel('No of recordings','interpreter','latex')
ylabel('$y$ axis','interpreter','latex')
title('Experiement 2','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
grid on
subplot(3,3,6)
plot(HO2(:,4),'r','LineWidth',1.5)
xlabel('No of recordings','interpreter','latex')
ylabel('$z$ axis','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
grid on
% Experiment 3
subplot(3,3,7)
plot(HO3(:,2),'g','LineWidth',1.5)
xlabel('No of recordings','interpreter','latex')
ylabel('$x$ axis','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
grid on
subplot(3,3,8)
plot(HO3(:,3),'g','LineWidth',1.5)
xlabel('No of recordings','interpreter','latex')
ylabel('$y$ axis','interpreter','latex')
title('Experiement 3','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
grid on
subplot(3,3,9)
plot(HO3(:,4),'g','LineWidth',1.5)
xlabel('No of recordings','interpreter','latex')
ylabel('$z$ axis','interpreter','latex')
set(gca,'TickLabelInterpreter','latex')
grid on
%------------------

%% Construct the experiement dataset

X = [HO1(:,2)'
    HO1(:,3)'
    HO1(:,4)'
    HO2(:,2)'
    HO2(:,3)'
    HO2(:,4)'
    HO3(:,2)'
    HO3(:,3)'
    HO3(:,4)'];

%extract the size of the dataset
[Xm,Xn] = size(X);

%% Compute the mean of all samples
 mn = mean(X,2); %mean(x,2) is the mean over the 2nd dimension, which means the rows.

%% Build the the mean-centring dataset (Data-mean) 
X = X - mn*ones(1,Xn);

%% Construct the covariance matrix L:
L = X*X'; %or L = cov(X)

%% Calculate the eigenvectors and the eigenvalues of the covariance matrix
[M,D] = eig(L); %Here: V are the eigenvectors and the diagonal elements of 
% D are the eigenvalues of the matrix L.

% extract the principal components eigenvalues, i.e.m diagonal elements of D
lambda1 = diag(D);

%% sort in decreasing order
[dummy,m_arrange] = sort(-1*lambda1);
lambda1 = lambda1(m_arrange);
M = M(:,m_arrange);

%% Get PCA ranks/scores/modes 
% This can be done by projecting our mean_centred dataset on the covariance
% matrix eigenvectors (principal component vector space)
Y1 = M'*X;

%% visulaise the scores/Modes
figure;
subplot(1,2,1)
plot(Y1(1,:),'b','LineWidth',1.5), hold on
plot(Y1(2,:),'r','LineWidth',1.5), hold on
plot(Y1(3,:),'g','LineWidth',1.5)
xlabel('Time/Frame','FontSize',14)
ylabel('Displacement Position','FontSize',14)
legend('Mode 1','Mode 2','Mode 3')
set(gca,'TickLabelInterpreter','latex')
axis square
grid on
hold on
%visualise the principal components
subplot(1,2,2)
plot(lambda1/sum(lambda1),'bo','linewidth',1.5)
xlabel('PCA Principal Components','FontSize',14)
ylabel('Variance in Percentage ','FontSize',14)
set(gca,'TickLabelInterpreter','latex')
axis square
grid on
%------------------

%% Singular Value Decomposition
%calculate the eignvalues and both eigenvectors
[U,S,V] = svd(X/sqrt(Xn-1),'econ');

%extract the diagonals, i.e., eigenvalues
lambda2= diag(S).^2;

%calculate the SVD modes, i.e., projections
Y2 = U'*X;

%visualise the first three modes
figure;
subplot(1,2,1)
plot(Y2(1,:),'b','LineWidth',1.5), hold on
plot(Y2(2,:),'r','LineWidth',1.5), hold on
plot(Y2(3,:),'g','LineWidth',1.5)
xlabel('Time/Frame','FontSize',14)
ylabel('Displacement Position','FontSize',14)
legend('Mode 1','Mode 2','Mode 3')
set(gca,'TickLabelInterpreter','latex')
grid on
axis square
hold on
%visualise the principal components
subplot(1,2,2)
plot(lambda2/sum(lambda2),'bo','linewidth',1.5)
xlabel('SVD Singular Values','FontSize',14)
ylabel('Variance in Percentage ','FontSize',14)
set(gca,'TickLabelInterpreter','latex')
grid on
axis square
%------------------


