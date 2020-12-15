clc 
clear
A=0.0154; %birth rate
Bh=0.09757382394; %transmission rate 
muh=0.019; %suseptible death rate 
sigmah=0.2; %infective removal rate either dead or cured
gammah=0.64; %Exposed turn into infected
fa=0.7;
M=0.07;
hrange=0:0.01:1;
brange= 0: 0.01: 1;
mum=0.028;
sigmam=0.0833;
Bm=0.11542507^2/Bh;
%for Bm= brange 
    %for Bh= hrange
    num=0;
figure(1)
while num<=20 
    num=num+1;
    para(num)=Avary;
     Avary=Avary*1.1
R0(num)=sqrt(muh*(sigmah+muh)*(gammah+muh)...
    *(sigmam+mum)*M*sigmah*Bh*Bm*Avary*sigmam)...
    /(muh*(sigmah+muh)*(gammah+muh)*(sigmam+mum)*mum);
plot(num,R0(num),'X');
hold on
if num>=2
changepara(num)=(para(num)-para(num-1))/para(num-1);
changeR0(num)=(R0(num)-R0(num-1))/R0(num-1);
end
end
%%
c=1:num-1;
changepara(changepara==0)=[];
changeR0(changeR0==0)=[];
figure()
plot(c,changepara,c,changeR0);