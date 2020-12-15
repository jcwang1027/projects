
clear; clc;
r=10^(-4);
alpha=10^(-4);
m=0.5;
s=1.2;
gamma=0.042;
sigma=0.5;

abslan=-2*r-sigma;

f=0.7;

arange = 0.2:0.01:0.34;  global beta    
k = 0; tspan = 0:0.1:1000;   
xmax = [];              
for beta = arange 
    
 f=@(t,x)[r+abslan-(r+abslan)*x(1)-abslan*x(2)-abslan*x(3)+alpha*x(1)*x(3)-beta*(x(1)^m)*(x(3)^s);
-(r+sigma)*x(2)+alpha*x(2)*x(3)+beta*(x(1)^m)*(x(3)^s);
sigma*x(2)-(r+gamma+alpha)*x(3)+alpha*x(3)^2];

    x0 = [0.8 0 0];                 
    k = k + 1; 
    [t,x] = ode45(f,tspan,x0);       
    count = find(t>100);  
    x = x(count,:); 
    j = 1; 
    n = length(x(:,1));  
    for i = 2:n-1 
       
         if abs(x(i-1,1)) < abs( x(i,1)) && abs(x(i,1)) > abs((x(i+1,1)))
            xmax(k,j)=x(i,1); 
            j=j+1; 
        end 
    end 
   
     if j>1 
        plot(beta,abs(xmax(k,1:j-1)),'k.'); 
     end
    hold on; 
    index(k)=j-1; 
end 
xlabel('Bifurcation parameter beta'); 
ylabel('y max'); 
title('Bifurcation diagram for beta'); 
