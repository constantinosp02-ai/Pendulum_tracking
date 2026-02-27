
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

%% Edge Detection: 1st order derivative

%detect the edges in the image using one of the 1st order derivative
%algorithm
%METHOD 1: Finds edges at those points where the gradient of I is maximum, 
% using the Prewitt approximation to the derivative.
ED1 = edge(Abw,'Prewitt');
%visualise it
subplot(2,3,4)
imshow(ED1)
colormap('gray')
title('1st Order Derivative Image: Prewitt')
set(gca,'XTick',[])
set(gca,'YTick',[])
%METHOD 2: Finds edges at those points where the gradient of the image I 
% is maximum, using the Sobel approximation to the derivative.
ED2 = edge(Abw,'Sobel');
subplot(2,3,5)
imshow(ED2)
colormap('gray')
title('1st Order Derivative Image: Sobel')
set(gca,'XTick',[])
set(gca,'YTick',[])
%% Edge Detection: 2nd order derivative

%detect the edges in the image using one of the 1st order derivative
%algorithm.
%Finds edges by looking for zero-crossings after filtering I with a 
% Laplacian of Gaussian (LoG) filter.
ED3 = edge(Abw,'log');
%visualise it
subplot(2,3,6)
imshow(ED3)
colormap('gray')
title('2nd Order Derivative Image')
set(gca,'XTick',[])
set(gca,'YTick',[])
%------------------------------------