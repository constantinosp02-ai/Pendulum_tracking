close all;
clear all;



%% Read RGB Image

%import image
A = imread('Belvoir Castle','jpeg'); %Glaxo Smith Kline Carbon Neutral Laboratory for Sustainable Chemistry

%show the coloured image
figure;
subplot(2,2,1)
imshow(A) 
title('Coloured Image')
set(gca,'XTick',[])
set(gca,'YTick',[])
%---------------------

%% Add Gaussian noise to the RGB image

%convert the image from uint8 to double
A2 = double(A);

%add Gaussian noise to the double-precisioned image
B_RGB = A2 + 100*randn(size(A2,1),size(A2,2),3);

%show the coloured noisy image
subplot(2,2,2)
imshow(uint8(B_RGB))
title('Noisy Coloured Image')
set(gca,'XTick',[])
set(gca,'YTick',[])
%---------------------

%% Convert the image from RGB to gray scale

%convert image to black and white
Abw = rgb2gray(A);

%show the grayscaled image
subplot(2,2,3)
imshow(Abw)
title('Grayscaled Image')
colormap('gray')
set(gca,'XTick',[])
set(gca,'YTick',[])
%---------------------

%% add Gaussian noise on the black/white image

%convert the BW image from image format to a matrix
Abw2 = double(Abw);

%add Gaussian noise to the double-precisioned BW image
B = Abw2 + 100*randn(size(Abw,1),size(Abw,2));

%plotting
subplot(2,2,4)
imshow(uint8(B))
title('Noisy BW Image')
colormap('gray')
set(gca,'XTick',[])
set(gca,'YTick',[])
%---------------------

%% Apply 2D Fourier Transofmr on the grasclaed image

%since pcolor flips thei mge upside down, let's flip it from the start
Abw2 = Abw2(size(Abw,1):-1:1,:);

%Fourier Transform
Abwt = fft2(Abw2);

%Shifting
Abwts = fftshift(Abwt);

%plotting
figure;
%show the grayscaled image
subplot(3,2,1)
imshow(Abw)
set(gca,'XTick',[])
set(gca,'YTick',[])
title('Grayscaled Image')
%show the FT of the BW image
subplot(3,2,2)
pcolor(log(abs(Abwts)))
shading interp
colormap('hot')
set(gca,'XTick',[])
set(gca,'YTick',[])
title('Fourier Trasnform of the Grascaled Image')
%--------------------

%% Apply 2D Fourier Transofmr on the noisy image

%Fourier Transform
Bt = fft2(B);

%Shifting
Bts = fftshift(Bt);

%plotting
%show the noisy grayscale imGE
subplot(3,2,3)
imshow(uint8(B))
colormap('gray')
set(gca,'XTick',[])
set(gca,'YTick',[])
title('Noisy BW Image')
%show the FT of the noisy BW image
subplot(3,2,4)
pcolor(log(abs(Bts)))
shading interp
colormap("gray")
set(gca,'XTick',[])
set(gca,'YTick',[])
title('Fourier Trasnform of the Noisy Image')
%--------------------

%% Filtering

%Define the wave number in 2D
kx = 1:size(Abw,2);
ky = 1:size(Abw,1);
[KX,KY] = meshgrid(kx,ky);

%middle of image width (a)
mid_width = size(Abw,2)/2;

%middle of image hiegith (b)
mid_height = size(Abw,1)/2;


% %First: Guassian filter (Bell filter)
% %define the filter width
% width = 0.0001; %the smaller the width the better the filtering
% %define the filter function
% F = exp(-width*(KX-mid_width).^2-width*(KY-mid_height).^2); 

%Second: Shannon filter (Square/Step filter)
% define the filter width
width = 200; %the smaller the width the better the filtering
% define the filter function
F = zeros(size(Abw,1),size(Abw,2)); 
Fx = mid_height+1-width:1:mid_height+1+width; %Or: (size(Abw,1)/2)+1-width:1:(size(Abw,1)/2)+1+width
Fy = mid_width+1-width:1:mid_width+1+width;
F(Fx,Fy) = ones(length(2*width+1),length(2*width+1));

%apply/convolving the filter on/with the transform shifted image
Btsf = Bts.*F; 

%shift it back to the space domain
Btf = ifftshift(Btsf);

%apply the inverse of Frourier Transform
Bf = ifft2(Btf);

%plotting
%the Fourier Transform of the filtered image
subplot(3,2,6)
pcolor(log(abs(Btsf)))
shading interp
colormap("gray");
set(gca,'XTick',[])
set(gca,'YTick',[])
title('Foruier Transform of the Filtered Image')
%the reconstructed image
subplot(3,2,5)
imshow(uint8(Bf)) %show it as an image
colormap("gray");
title('Reconstructed Filtered Image')
%--------------------
