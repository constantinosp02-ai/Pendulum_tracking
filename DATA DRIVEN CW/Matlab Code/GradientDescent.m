
%Gradient Decent Algorithm

%Gradient descent is an optimization algorithm which is commonly-used to 
% train machine learning models and neural networks. It trains machine 
% learning models by minimizing errors between predicted and actual results.

%Gradient descent is an algorithm that numerically estimates where a 
% function outputs its lowest values.

% The idea is to take repeated steps in the opposite direction of the 
% gradient (or approximate gradient) of the function at the current point, 
% because this is the direction of steepest descent.


clear all;
close all;

%define the dimensions of the function
x = -4:0.1:4;
y = -4:0.1:4;

%build an xy mesh
[X,Y] = meshgrid(x,y);

%define the quadratic form of the Ax=b linear system of equation, i.e., the
%surface that defines it. 
fsurf = X.^2+3.*Y.^2;

%show the surface funciton along with its contour plots
figure;
surf(X,Y,fsurf,'FaceAlpha',0.6)
colormap gray
shading interp
hold on
contour(X,Y,fsurf,'LineWidth',1.5)
xlabel('$x$','interpreter','latex')
ylabel('$y$','interpreter','latex')
zlabel('$f(x,y)=x^2+3y^2$','interpreter','latex')
set(gca,'TickLabelInterpreter','latex','FontSize',12)
view(-73,36)
hold on

%clear the varibles x and y to reuse them again
clear x
clear y

%suggest random initial values for both variables
x(1)= 2.5;
y(1) = 2;
%calculate the initial funciton, i.e., the point that corresponds to the
%random initial point
f(1) = x(1).^2+3.*y(1).^2;
%plot the first random point on the surface
plot3(x,y,f,'ro','LineWidth',3)

%Iterate until the minimum of the surface is reached
for j=1:100
    %define the jump step
    tau = (x(j)^2+9*y(j)^2)/(2*x(j)^2+54*y(j)^2);
    %caluclate the future steps
    x(j+1) = (1-2*tau)*x(j);
    y(j+1) = (1-6*tau)*y(j);
    f(j+1) = x(j+1).^2+3.*y(j+1).^2;
    %show the future points on the surface figure 
    figure(1)
    hold on
    plot3(x(j+1),y(j+1),f(j+1),'ro','LineWidth',3)
    pause(2)
    %if the tolerance between each points or the maximum number
    %of iteractions are achieved, exit the iteration cycle.
    if abs(f(j+1)-f(j)) < 10^(-6)
        break
    end
end

%print the solution of the linear system of equation, i.e., the values of
%the variables x and y
x(end)
y(end)





