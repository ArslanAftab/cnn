%SGD+monmenton
clc
clear
output=[]
%define initial value for v(t)
momentum_1_01=0;
momentum_1_02=0;
momentum_1_03=0;
momentum_1_11=0;
momentum_1_12=0;
momentum_1_13=0;
momentum_2_11=0;
momentum_2_21=0;
momentum_2_31=0;
rho=0.9;%define rho value
%define points and desired output
x=-1:0.05:1;
len=length(x);
x_0=1;
d=0.8*x.^3+0.3*x.^2-0.4*x+normrnd(0,0.02,[1,len]);
figure,plot(x,d,'k+');
%initial setup(weight, LR and epochs)
w_1_01=rand;
w_1_02=rand;
w_1_03=rand;
w_1_11=rand;
w_1_12=rand;
w_1_13=rand;
w_2_11=rand;
w_2_21=rand;
w_2_31=rand;
learning_rate=0.01;
epochs=10000;
for i=1:epochs
for j=1:41
v_1_1=x_0*w_1_01+x(j)*w_1_11;
v_1_2=x_0*w_1_02+x(j)*w_1_12;
v_1_3=x_0*w_1_03+x(j)*w_1_13;
%using tanh in hidden layer
x_1_out1=(exp(v_1_1)-exp(-v_1_1))/(exp(v_1_1)+exp(-v_1_1)); %X_out of first neutron in first layer
x_1_out2=(exp(v_1_2)-exp(-v_1_2))/(exp(v_1_2)+exp(-v_1_2));%X_out of second neutron in first layer
x_1_out3=(exp(v_1_3)-exp(-v_1_3))/(exp(v_1_3)+exp(-v_1_3));%X_out of third neutron in first layer
v_2_1=x_1_out1*w_2_11+x_1_out2*w_2_21+x_1_out3*w_2_31;
%using pure linear in output layer
if v_2_1<0
    x_2_out1=v_2_1; dphidv_2_1=1;
else
    x_2_out1=v_2_1; dphidv_2_1=1;
end


%weight derivation in output layer
delta_2_1=(d(j)-x_2_out1)*dphidv_2_1;%local gradient

dldw_2_11=delta_2_1*x_1_out1;
dldw_2_21=delta_2_1*x_1_out2;
dldw_2_31=delta_2_1*x_1_out3;

momentum_2_11=rho*momentum_2_11+dldw_2_11;
momentum_2_21=rho*momentum_2_21+dldw_2_21;
momentum_2_31=rho*momentum_2_31+dldw_2_31;

w_2_11=w_2_11+learning_rate*momentum_2_11;
w_2_21=w_2_21+learning_rate*momentum_2_21;
w_2_31=w_2_31+learning_rate*momentum_2_31;

%hidden layer derivation
delta_1_1=delta_2_1*w_2_11*(1-(x_1_out1)^2);%local gradient
delta_1_2=delta_2_1*w_2_21*(1-(x_1_out2)^2);%local gradient
delta_1_3=delta_2_1*w_2_31*(1-(x_1_out3)^2);%local gradient

dldw_1_01=delta_1_1*x_0;
dldw_1_02=delta_1_2*x_0;
dldw_1_03=delta_1_3*x_0;
dldw_1_11=delta_1_1*x(j);
dldw_1_12=delta_1_2*x(j);
dldw_1_13=delta_1_3*x(j);

momentum_1_01=rho*momentum_1_01+dldw_1_01;
momentum_1_02=rho*momentum_1_02+dldw_1_02;
momentum_1_03=rho*momentum_1_03+dldw_1_03;
momentum_1_11=rho*momentum_1_11+dldw_1_11;
momentum_1_12=rho*momentum_1_12+dldw_1_12;
momentum_1_13=rho*momentum_1_13+dldw_1_13;


w_1_01=w_1_01+learning_rate*momentum_1_01;
w_1_02=w_1_02+learning_rate*momentum_1_02;
w_1_03=w_1_03+learning_rate*momentum_1_03;
w_1_11=w_1_11+learning_rate*momentum_1_11;
w_1_12=w_1_12+learning_rate*momentum_1_12;
w_1_13=w_1_13+learning_rate*momentum_1_13;

end
end

x=-0.97:0.1:0.93;
for k=1:length(x)
v_1_1=x_0*w_1_01+x(k)*w_1_11;
v_1_2=x_0*w_1_02+x(k)*w_1_12;
v_1_3=x_0*w_1_03+x(k)*w_1_13;
%using tanh in hidden layer
x_1_out1=(exp(v_1_1)-exp(-v_1_1))/(exp(v_1_1)+exp(-v_1_1)); %X_out of first neutron in first layer
x_1_out2=(exp(v_1_2)-exp(-v_1_2))/(exp(v_1_2)+exp(-v_1_2));%X_out of second neutron in first layer
x_1_out3=(exp(v_1_3)-exp(-v_1_3))/(exp(v_1_3)+exp(-v_1_3));%X_out of third neutron in first layer
v_2_1=x_1_out1*w_2_11+x_1_out2*w_2_21+x_1_out3*w_2_31;
%using linear in output layer
if v_2_1<0
    x_2_out1=v_2_1; dphidv_2_1=1;
else
    x_2_out1=v_2_1; dphidv_2_1=1;
end
output=[output x_2_out1];
end
hold on
plot(x,output,'r-')
xlabel('x')
ylabel('output')
legend('training data','SGD+Momentum test prediction')
