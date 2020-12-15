 for betai=0.05:0.01:0.25;
betad=0.71;%10 to 20 percentage
Ti=6;
Te=11;
Td=2;%2,4,6 three senerio
f=0.7;
f=@(t,x)[-betai*x(1)*x(3)/(x(1)+x(2)+x(3)+x(4))-betad*x(1)*x(5)/(x(1)+x(2)+x(3)+x(4));
betai*x(1)*x(3)/(x(1)+x(2)+x(3)+x(4))+betad*x(1)*x(5)/(x(1)+x(2)+x(3)+x(4))-x(2)/Te
x(2)/Te-x(3)/Ti;
(1-f)*x(3)/Ti;
f*x(3)/Ti-x(5)/Td];
t=linspace(1,1000,100);
sol=ode113(f,[0,1000],[0.8,0.1,0.1,0,0]);
[y1,dy1]=deval(sol,t);
figure(1)
plot(t,y1(1,:));

plot(t,y1(2,:));

plot(t,y1(3,:));

plot(t,y1(4,:));

plot(t,y1(5,:));
sum(y1);
 end