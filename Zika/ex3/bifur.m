clear; clc;
A=0.0154; %birth rate
%Bh=0.9757382394; %transmission rate 
muh=0.019; %suseptible death rate 
sigmah=0.2; %infective removal rate either dead or cured
gammah=0.64; %Exposed turn into infected
fa=0.7;
M=0.07;
hrange=0:0.01:1;
brange= 0: 0.01: 1;
mum=0.028;
sigmam=0.0833;
Bm=0.3;
 global Bh; 
    
k = 0; tspan = 0:2000;     
xmax = [];       figure(1)      
for Bh = brange 
    f=@(t,x)[A-Bh*x(7)*x(1)-muh*x(1);
    Bh*x(7)*x(1)-(muh+sigmah)*x(2);
    sigmah*x(2)-(gammah+muh)*x(3);
    (1-fa)*gammah*x(3)-muh*x(4);
    M-Bm*x(3)*x(5)-mum*x(5);
Bm*x(3)*x(5)-(sigmam+mum)*x(6);
sigmam*x(6)-mum*x(7)];
    x0 = [0.08;0.01;0.01;0.1;0.8;0.01;0.01];              
    k = k + 1; 
    [t,x] = ode45(f,tspan,x0); 

    count = find(t>100);  
    x = x(count,:); 
    j = 1; 
    n = length(x(:,1));  
    for i = 2:n-1 
       
          if (x(i-1,1)) < ( x(i,1)) && (x(i,1)) > ((x(i+1,1)))
            xmax(k,j)=x(i,1); 
            j=j+1; 
          end
    end 
    
    if j>1 
        
        plot(Bh,xmax(k,1:j-1),'k.'); 
    end 
    hold on; 
    index(k)=j-1; 
end 
xlabel('Bifurcation parameter Bh'); 
ylabel('y max');
title('Bifurcation diagram for Bh'); 
t1=1:1900;
figure(2)
plot(t1,x(:,1));
hold on
grid on
plot(t1,x(:,2));
hold on
plot(t1,x(:,3));
hold on
plot(t1,x(:,4));
hold on
plot(t1,x(:,5));
hold on
plot(t1,x(:,6));
hold on
plot(t1,x(:,7));
