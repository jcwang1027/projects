clc 
clear
A=0.0154; %birth rate
Bh=0.08885; %transmission rate 
muh=0.019; %suseptible death rate 
sigmah=0.2; %infective removal rate either dead or cured
gammah=0.64; %Exposed turn into infected
fa=0.01;
M=0.07;
mum=0.01806095170;
sigmam=0.0833;
Bm=0.116;
time=100;
brange=0:0.01:0.8;
hrange=0:0.01:1;
%for mum= brange 

f=@(t,x)[A-Bh*x(7)*x(1)-muh*x(1);
    Bh*x(7)*x(1)-(muh+sigmah)*x(2);
    sigmah*x(2)-(gammah+muh)*x(3);
    (1-fa)*gammah*x(3)-muh*x(4);
    M-Bm*x(3)*x(5)^0.5-mum*x(5);
Bm*x(3)*x(5)-(sigmam+mum)*x(6);
sigmam*x(6)-mum*x(7)];

t1=linspace(1,10000,time);
%sol=ode45(f,t1,[0 0 0 0 0 0 0]);
%sol=ode45(f,t1,[1000;200;100;500;800;100;300]);
sol=ode45(f,t1,[0.8;0.1;0.05;0.05;0.8;0.1;0.1]);
[y1,dy1]=deval(sol,t1);

figure(1)
plot(t1,y1(1,:),'.');
hold on
grid on
plot(t1,y1(2,:));
hold on
plot(t1,y1(3,:));
hold on
plot(t1,y1(4,:));
hold on
plot(t1,y1(5,:),'.');
hold on
plot(t1,y1(6,:));
hold on
plot(t1,y1(7,:));
legend('Suceptible human','Exposed human','Infected human','Recovered human','Suceptible mosquito','Exposed mosquito','Im');
xlabel('Time t');
ylabel('Portion of total population');
grid on
title('Basic Reproduction number less than 1')
i=2:4;
j=6:7;
s1=sum(y1(i,:));
s2=sum(y1(j,:));
figure(2)
c=linspace(1,time,time);
plot(c,s1,c,s2);
xlabel('Time t');
ylabel('Portion of total population');
legend('Sum of Exposed and Infected human','Sum of Exposed and Infected mosquitoes');
grid on
title('Basic Reproduction number less than 1')
R0=sqrt(muh*(sigmah+muh)*(gammah+muh)...
    *(sigmam+mum)*M*sigmah*Bh*Bm*A*sigmam)...
    /(muh*(sigmah+muh)*(gammah+muh)*(sigmam+mum)*mum)

 y1(2,100)+y1(3,100)+y1(7,100)+y1(6,100)