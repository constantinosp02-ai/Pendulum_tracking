


close all;
clear all;

%% Read RGB Image

%import image
A = imread('Belvoir Castle','jpeg'); %Glaxo Smith Kline Carbon Neutral Laboratory for Sustainable Chemistry

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
subplot(2,3,2)
imshow(Abw)
colormap('gray')
title('Grayscaled Image')
set(gca,'XTick',[])
set(gca,'YTick',[])
%---------------------

%% Fourier Transform on the BW photo

%convert the BW image from image format to a matrix
Abw = double(Abw);

%Fourier Transform
At = fft2(Abw);

%shifting
Abwt = abs(fftshift(At));

%plotting
subplot(2,3,3)
pcolor(log(Abwt))
axis square
shading interp
colormap('hot')
title('Foruier Transform')
set(gca,'XTick',[])
set(gca,'YTick',[])
%---------------------

%% Apply image compression: Fourier Transform (FT)

% look for  the small frequencies and zeroing them out

%identify the size of the image
[nx,ny] = size(Abw);

%set a counter
count_pic = 4;

%go through several cut-off thresholds
for threshold = 0.1*[0.001 0.005 0.01] * max(max(abs(At)))
    %extract the frequncies that are above the current threshold
    ind = abs(At) > threshold;
    %zero out all other frequencies
    Atlow = At.*ind;
    %compute how many frequencies are thrown away
    count = nx*ny - sum(sum(ind));
    %calculate the poercentage of the kept frequencies
    percent = 100 - count/(nx*ny)*100;
    %convert it to an image type
    Alow = uint8(ifft2(Atlow));
    %show it
    figure(1)
    subplot(2,3,count_pic)
    imshow(Alow)
    axis square
    %go tho the second loop/threshold
    count_pic = count_pic+1;

    drawnow
    title([num2str(percent) '% of FFT basis'])
end

%% plot the 2D plot of the image frequencies
figure(2)
%original image
subplot(1,2,1)
surf(fliplr(Abw));
shading interp
axis square
xlabel('Width')
ylabel('Height')
zlabel('Frequency')
title('Uncompressed Image')
colormap(jet)
set(gca,'TickLabelInterpreter','latex')
%compressed image
subplot(1,2,2)
surf(fliplr(Alow));
shading interp
xlabel('Width')
ylabel('Height')
zlabel('Frequency')
axis square
title('Compressed Image')
colormap(jet)
colorbar
set(gca,'TickLabelInterpreter','latex')


% %% create a video that rotates sideways and up-down
% fig = figure(2);
% clf
% 
% % --- Original image ---
% subplot(1,2,1)
% surf(fliplr(Abw));
% shading interp
% axis square
% xlabel('Width')
% ylabel('Height')
% zlabel('Frequency')
% title('Uncompressed Image')
% colormap(jet)
% set(gca,'TickLabelInterpreter','latex')
% 
% % --- Compressed image ---
% subplot(1,2,2)
% surf(fliplr(Alow));
% shading interp
% axis square
% xlabel('Width')
% ylabel('Height')
% zlabel('Frequency')
% title('Compressed Image')
% colormap(jet)
% % colorbar
% set(gca,'TickLabelInterpreter','latex')
% 
% v = VideoWriter('compression_rotation.mp4','MPEG-4');
% v.FrameRate = 5;   % smooth motion
% open(v);
% 
% 
% for az = 0:2:360
%     subplot(1,2,1)
%     view(az,30)
% 
%     subplot(1,2,2)
%     view(az,30)
% 
%     drawnow
%     writeVideo(v, getframe(fig));
% end
% 
% 
% for el = 10:1:100
%     subplot(1,2,1)
%     view(180,el)
% 
%     subplot(1,2,2)
%     view(180,el)
% 
%     drawnow
%     writeVideo(v, getframe(fig));
% end
% 
% close(v);