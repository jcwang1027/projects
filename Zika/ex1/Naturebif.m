clear; clc;
   
betad=0.71;%10 to 20 percentage
Ti=6;
Te=11;
Td=2;%2,4,6 three senerio
f=0.7;
global betai; 
arange = 0.05:0.001:0.25;       % Range for parameter 
k = 0; tspan = 1:0.1:200;      % Time interval for solving
xmax = [];              % A matrix for storing the sorted value of x1
for betai = arange 
    
f=@(t,x)[-1*betai*x(1)*x(3)/(x(1)+x(2)+x(3)+x(4))-betad*x(1)*x(5)/(x(1)+x(2)+x(3)+x(4));
betai*x(1)*x(3)/(x(1)+x(2)+x(3)+x(4))+betad*x(1)*x(5)/(x(1)+x(2)+x(3)+x(4))-x(2)/Te
x(2)/Te-x(3)/Ti;
(1-f)*x(3)/Ti;
f*x(3)/Ti-x(5)/Td]; 
   
    k = k + 1; 
    [t,x] = ode113(f,tspan,[0.8,0.1,0.1,0,0]);
   
    count = find(t>100);  % find all the t_values which is >100 
    x = x(count,:); 
    j = 1; 
    n = length(x(:,1));  % find the length of vector x1(x in our problem)
    for i = 2:n-1 
        % check for the min value in 1st column of sol matrix
        if (x(i-1,1)+eps) < x(i,1) && x(i,1) > (x(i+1,1)+eps)
            xmax(k,j)=x(i,1); % Sorting the values of x1 in increasing order
            j=j+1; 
        end 
    end 
    % generating bifurcation map by plotting j-1 element of kth row each time 
    if j>1 
        plot(betai,xmax(k,1:j-1),'k.'); 
    end 
    hold on; 
    index(k)=j-1; 
end 
xlabel('Bifurcation parameter a'); 
ylabel('y max'); 
title('Bifurcation diagram for a'); 