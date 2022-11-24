
clc
clear
tic;
output=[]
%define points and desired output
x=-1:0.05:1;
len=length(x);
x_0=1;
rng(0,'twister');
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
w_2_12=rand;
w_2_13=rand;
w_2_21=rand;
w_2_22=rand;
w_2_23=rand;
w_2_31=rand;
w_2_32=rand;
w_2_33=rand;
w_3_11=rand;
w_3_21=rand;
w_3_31=rand;
learning_rate=0.01;
epochs=20000;
for i=1:epochs
    for j=1:41
        %using tanh in hidden layers
        v_1_1=x_0*w_1_01+x(j)*w_1_11;
        v_1_2=x_0*w_1_02+x(j)*w_1_12;
        v_1_3=x_0*w_1_03+x(j)*w_1_13;
        x_1_out1=(exp(v_1_1)-exp(-v_1_1))/(exp(v_1_1)+exp(-v_1_1));%X_out of first neutron in first layer
        x_1_out2=(exp(v_1_2)-exp(-v_1_2))/(exp(v_1_2)+exp(-v_1_2));%X_out of second neutron in first layer
        x_1_out3=(exp(v_1_3)-exp(-v_1_3))/(exp(v_1_3)+exp(-v_1_3));%X_out of third neutron in first layer
        v_2_1=x_1_out1*w_2_11+x_1_out2*w_2_21+x_1_out3*w_2_31;
        v_2_2=x_1_out1*w_2_12+x_1_out2*w_2_22+x_1_out3*w_2_32;
        v_2_3=x_1_out1*w_2_13+x_1_out2*w_2_23+x_1_out3*w_2_33;
        x_2_out1=(exp(v_2_1)-exp(-v_2_1))/(exp(v_2_1)+exp(-v_2_1));%X_out of first neutron in second layer
        x_2_out2=(exp(v_2_2)-exp(-v_2_2))/(exp(v_2_2)+exp(-v_2_2));%X_out of second neutron in second layer
        x_2_out3=(exp(v_2_3)-exp(-v_2_3))/(exp(v_2_3)+exp(-v_2_3));%X_out of third neutron in second layer
        v_3_1=x_2_out1*w_3_11+x_2_out2*w_3_21+x_2_out3*w_3_31;
        %using pure linear in output layer
        x_3_out1=v_3_1;
        dphidv_3_1=1;


        delta_3_1=(d(j)-x_3_out1)*dphidv_3_1;%local gradient
f
        %weight derivation in output layer
        w_3_11=w_3_11+learning_rate*delta_3_1*x_2_out1;
        w_3_21=w_3_21+learning_rate*delta_3_1*x_2_out2;
        w_3_31=w_3_31+learning_rate*delta_3_1*x_2_out3;

        %hidden layer derivation
        delta_2_1=delta_3_1*w_3_11*(1-(x_2_out1)^2);
        delta_2_2=delta_3_1*w_3_21*(1-(x_2_out2)^2);
        delta_2_3=delta_3_1*w_3_31*(1-(x_2_out3)^2);
        w_2_11=w_2_11+learning_rate*delta_2_1*x_1_out1;
        w_2_12=w_2_12+learning_rate*delta_2_2*x_1_out1;
        w_2_13=w_2_13+learning_rate*delta_2_3*x_1_out1;
        w_2_21=w_2_21+learning_rate*delta_2_1*x_1_out2;
        w_2_22=w_2_22+learning_rate*delta_2_2*x_1_out2;
        w_2_23=w_2_23+learning_rate*delta_2_3*x_1_out2;
        w_2_31=w_2_31+learning_rate*delta_2_1*x_1_out3;
        w_2_32=w_2_32+learning_rate*delta_2_2*x_1_out3;
        w_2_33=w_2_33+learning_rate*delta_2_3*x_1_out3;

        delta_1_1=(delta_2_1*w_2_11+delta_2_2*w_2_12+delta_2_3*w_2_13)*(1-(x_1_out1)^2);
        delta_1_2=(delta_2_1*w_2_21+delta_2_2*w_2_22+delta_2_3*w_2_23)*(1-(x_1_out2)^2);
        delta_1_3=(delta_2_1*w_2_31+delta_2_2*w_2_32+delta_2_3*w_2_33)*(1-(x_1_out3)^2);
        w_1_01=w_1_01+learning_rate*delta_1_1*x_0;
        w_1_02=w_1_02+learning_rate*delta_1_2*x_0;
        w_1_03=w_1_03+learning_rate*delta_1_3*x_0;
        w_1_11=w_1_11+learning_rate*delta_1_1*x(j);
        w_1_12=w_1_12+learning_rate*delta_1_2*x(j);
        w_1_13=w_1_13+learning_rate*delta_1_3*x(j);

    end
end

x=-0.97:0.1:0.93;
for k=1:length(x)
v_1_1=x_0*w_1_01+x(k)*w_1_11;
v_1_2=x_0*w_1_02+x(k)*w_1_12;
v_1_3=x_0*w_1_03+x(k)*w_1_13;
%using tanh in hidden layer
v_1_1=x_0*w_1_01+x(k)*w_1_11;
v_1_2=x_0*w_1_02+x(k)*w_1_12;
v_1_3=x_0*w_1_03+x(k)*w_1_13;
x_1_out1=(exp(v_1_1)-exp(-v_1_1))/(exp(v_1_1)+exp(-v_1_1));%X_out of first neutron in first layer
x_1_out2=(exp(v_1_2)-exp(-v_1_2))/(exp(v_1_2)+exp(-v_1_2));%X_out of second neutron in first layer
x_1_out3=(exp(v_1_3)-exp(-v_1_3))/(exp(v_1_3)+exp(-v_1_3));%X_out of third neutron in first layer
v_2_1=x_1_out1*w_2_11+x_1_out2*w_2_21+x_1_out3*w_2_31;
v_2_2=x_1_out1*w_2_12+x_1_out2*w_2_22+x_1_out3*w_2_32;
v_2_3=x_1_out1*w_2_13+x_1_out2*w_2_23+x_1_out3*w_2_33;
x_2_out1=(exp(v_2_1)-exp(-v_2_1))/(exp(v_2_1)+exp(-v_2_1));%X_out of first neutron in second layer
x_2_out2=(exp(v_2_2)-exp(-v_2_2))/(exp(v_2_2)+exp(-v_2_2));%X_out of second neutron in second layer
x_2_out3=(exp(v_2_3)-exp(-v_2_3))/(exp(v_2_3)+exp(-v_2_3));%X_out of third neutron in second layer
v_3_1=x_2_out1*w_3_11+x_2_out2*w_3_21+x_2_out3*w_3_31;
%using linear in output layer
x_3_out1=v_3_1;
dphidv_3_1=1;
output=[output x_3_out1];
end
hold on
plot(x,output,'r-')
xlabel('x')
ylabel('output')
legend('training data','structure 1-3-3-1 test prediction')
toc;
