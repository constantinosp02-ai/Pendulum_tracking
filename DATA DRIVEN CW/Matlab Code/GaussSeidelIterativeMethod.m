
%This code implements Gauss Seidel iterarive method to solve Ax=b system of
%equations

% In numerical linear algebra, the Gauss–Seidel method, also known as the 
% Liebmann method or the method of successive displacement, is an 
% iterative method used to solve a system of linear equations.

% In mathematics, a square matrix is said to be diagonally dominant if, 
% for every row of the matrix, the magnitude/absolute value of the 
% diagonal entry in a row is larger than or equal to the sum of the 
% magnitudes/absolute values of all the other (non-diagonal) entries in 
% that row. 
%--------------

close all;
clear all;
clc

%Equations to solve (Strictly Dominant Diagonal A Matrix (SDD))
% 4x-y+z=7
% 4x-8y+z=-21
% -2x+y+5z=15

%Define initial guess for the three variables
x1(1) = 13;
y1(1) = -5;
z1(1) = pi;

%go through iterations
for j = 1:10
    x1(j+1) = (7+y1(j)-z1(j))/4;
    y1(j+1) = (21+4*x1(j+1)+z1(j))/8;
    z1(j+1) = (15+2*x1(j+1)-y1(j+1))/5;
end

%plot the solution
figure;
% subplot(1,2,1)
plot(1:11,x1,'b',1:11,y1,'r',1:11,z1,'g','LineWidth',1.5)
xlabel('Number of Iterations','Interpreter','latex')
ylabel('Solutions','Interpreter','latex')
title('Matrix $A$ is SDD','Interpreter','latex')
set(gca,'TickLabelInterpreter','latex','FontSize',12)
axis square
hold on

%------------------------

% %Equatons to solve (NOT Strictly Dominant Diagonal A Matrix (SDD))
% % -2x+y+5z=15
% % 4x-8y+z=-21
% % 4x-y+z=7
% 
% %Define initial guess for the three variables
% x2(1) = 13;
% y2(1) = -5;
% z2(1) = pi;
% 
% %go through iterations
% for j = 1:10
%     x2(j+1) = (-15+y2(j)+5*z2(j))/2;
%     y2(j+1) = (21+4*x2(j+1)+z2(j))/8;
%     z2(j+1) = 7-4*x2(j+1)+y2(j+1);
% end
% 
% %plot the solution
% subplot(1,2,2)
% plot(1:11,x2,'b',1:11,y2,'r',1:11,z2,'g','LineWidth',1.5)
% xlabel('Number of Iterations','Interpreter','latex')
% ylabel('Solutions','Interpreter','latex')
% title('Matrix $A$ is not SDD','Interpreter','latex')
% set(gca,'TickLabelInterpreter','latex','FontSize',12)
% axis square
%------------------------

%comparision between Jacobi method and Gauss-Seidel method
%Define initial guess for the three variables
x3(1) = 13;
y3(1) = -5;
z3(1) = pi;

%go through Jacobi iterations
for j = 1:10
    x3(j+1) = (7+y3(j)-z3(j))/4;
    y3(j+1) = (21+4*x3(j)+z3(j))/8;
    z3(j+1) = (15+2*x3(j)-y3(j))/5;
end

%plot both solutions
figure;
subplot(1,2,2)
%Guass-Seidel
plot(1:11,x1,'b',1:11,y1,'r',1:11,z1,'g','LineWidth',1.5)
xlabel('Number of Iterations','Interpreter','latex')
ylabel('Solutions','Interpreter','latex')
legend('$x$','$y$','$z$','Interpreter','latex')
title('Gauss-Seidel Method','Interpreter','latex')
set(gca,'TickLabelInterpreter','latex','FontSize',12)
axis square
hold on
%Jacobi
subplot(1,2,1)
plot(1:11,x3,'b',1:11,y3,'r',1:11,z3,'g','LineWidth',1.5)
xlabel('Number of Iterations','Interpreter','latex')
ylabel('Solutions','Interpreter','latex')
legend('$x$','$y$','$z$','Interpreter','latex')
title('Jacobi Method','Interpreter','latex')
set(gca,'TickLabelInterpreter','latex','FontSize',12)
axis square
%------------------------


