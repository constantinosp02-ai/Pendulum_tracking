

clear all;
close all;

%% Read RGB Image

%import image
A = imread('Belvoir Castle','jpeg'); %Glaxo Smith Kline Carbon Neutral Laboratory for Sustainable Chemistry

%define image dimensions
nx = size(A,1);
ny = size(A,2);

%show the coloured image
figure;
subplot(2,3,1)
imshow(A) 
title('Coloured Image')
set(gca,'XTick',[])
set(gca,'YTick',[])
%---------------------

%% Convert the image from RGB to gray scale

%convert image to black and white
Abw = rgb2gray(A);

%show the grayscaled image
subplot(2,3,3)
imshow(Abw)
colormap('gray')
title('Grayscaled Image')
set(gca,'XTick',[])
set(gca,'YTick',[])
%---------------------

%% Image Compression: Singular Value Decomposition (SVD)

%convert the grayscaled image from image type to double type
X = double(Abw);

%Apply SVD
[U,S,V] = svd(X,'econ');

%set the plotting counter
plotind = 4;
%go through several cut-off truncations
for r = [5 20 100] %truncation value
    %calculate the reduced-order image
    Xapprox = U(:,1:r)*S(1:r,1:r)*V(:,1:r)'; %approximated image
    %call the subplot with the current counter 
    subplot(2,3,plotind)
    %go through the next subplot
    plotind = plotind + 1;
    % show the approximated image
    imagesc(Xapprox)
    axis off
    axis square
    title(['r= ',num2str(r)]);
end

%to show eigenvalues entire spectrum
figure
subplot(1,2,1)
lambda = diag(S);
semilogy(lambda,'ko','LineWidth',2)
hold on
plot(lambda(1:50),'ro','LineWidth',1.5)
xlabel('Number of Eigenvalues, $r$','interpreter','latex')
ylabel('Magnitude of Eigenvalues','interpreter','latex')
grid on
axis square
% xlim([-50 1550])
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
%Cumulative Energy
subplot(1,2,2)
svdcumsum = cumsum(diag(S))/sum(diag(S));
plot(svdcumsum,'ko','LineWidth',2)
hold on
plot(svdcumsum(1:50),'ro','LineWidth',1.5)
xlabel('Number of Eigenvalues, $r$','interpreter','latex')
ylabel('Cumulative Energy','interpreter','latex')
grid on
axis square
% xlim([-50 1550])
% ylim([0 1.1])
set(gca,'TickLabelInterpreter','latex')
set(gca,'FontSize',16)
% the first column in U and V accounts for 28% of the total information  in 
% the dog.
%------------------------------------