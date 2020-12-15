
clc;
clear
% 
% Parameter Value
r=10^(-4);
m=0.5;
s=1.2;
gamma=0.042;
sigma=0.5;
alpha=10^(-4);
abslan=10^(-3);
beta=0.28;

% Differential Equation
f=@(t,x)[r+abslan-(r+abslan)*x(1)-abslan*x(2)-abslan*x(3)+alpha*x(1)*x(3)-beta*(x(1)^m)*(x(3)^s);
-(r+sigma)*x(2)+alpha*x(2)*x(3)+beta*(x(1)^m)*(x(3)^s);
sigma*x(2)-(r+gamma+alpha)*x(3)+alpha*x(3)^2];
t1=linspace(1,10000,1000);


sol=ode45(f,t1,[0.7,0.2,0.1]);
[y1,dy1]=deval(sol,t1);
d=(1-(y1(1,:)+y1(2,:)+y1(3,:)));

%[y1,dy1]=deval(sol,t);
figure(1)
plot(t1,(y1(1,:)));
hold on
plot(t1,(y1(2,:)));
hold on
plot(t1,(y1(3,:)));
hold on
plot(t1,(1-(y1(1,:)+y1(2,:)+y1(3,:))));
legend('Suceptible human','Exposed human','Infected human','Recovered human')
xlabel('Time t');
ylabel('Portion of total population');
grid on
title('disease free equilibrium')
figure(2)
plot(dy1(1,200:300),dy1(3,200:300),'-');
hold on
plot(dy1(1,300:400),dy1(3,300:400),'-');
hold on
plot(dy1(1,400:500),dy1(3,400:500),'-');
hold on
plot(dy1(1,500:600),dy1(3,500:600),'-');
hold on
plot(dy1(1,600:700),dy1(3,600:700),'-');
legend('Time_1','Time_2','Time_3','Time_4','Time_5');
xlabel('Suceptible population');
ylabel('Infected population');
grid on
title('Phase Portrait for stable focus');
