clear all;
close all;
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Import Data and Randomise Order %
% Choose 70% as training set      %
% And remaining 30% as test set   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Change iris types into numbers
% Setosa = 0, Versicolor = 1, Virginica = 2
irisData = readtable('IrisData.txt');
irisData.Var5 = categorical(irisData.Var5);
irisData.Var5 = renamecats(irisData.Var5, {'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'}, {'0', '1','2'});
irisData.Var5 = str2double(string(irisData.Var5));
irisData = table2array(irisData);

% Randomize order
[no_samples,m] = size(irisData);
i = randperm(no_samples);
irisDataJumbled = irisData(i,:);

% Split into training set and test set
SeventyPercent = round(0.7*no_samples,0);
IrisDataTrain = irisDataJumbled(1:SeventyPercent,:);
IrisDataTest = irisDataJumbled(SeventyPercent+1:no_samples,:);

% initial setup weights w_{layer}_{node} 
% index into array to get weight, i.e. 
% w_11_(2) means weight from layer 1: node 1 to layer 2: node 2

% input layer 
w_11 = rand(5,1); % node 1 weights
w_12 = rand(5,1); % node 2 weights
w_13 = rand(5,1); % node 3 weights
w_14 = rand(5,1); % node 4 weights
% first hidden layer
w_21 = rand(3,1); % node 1 weights
w_22 = rand(3,1); % node 2 weights
w_23 = rand(3,1); % node 3 weights
w_24 = rand(3,1); % node 4 weights
w_25 = rand(3,1); % node 5 weights
% second hidden layer
w_31 = rand(3,1); % node 1 weights
w_32 = rand(3,1); % node 2 weights
w_33 = rand(3,1); % node 3 weights

% initial setup biases b_{layer}_{node}
% first hidden layer
b_2_1 = rand(); % node 1 bias
b_2_2 = rand(); % node 2 bias
b_2_3 = rand(); % node 3 bias
b_2_4 = rand(); % node 4 bias
b_2_5 = rand(); % node 5 bias

% second hidden layer
b_3_1 = rand(); % node 1 bias
b_3_2 = rand(); % node 2 bias
b_3_3 = rand(); % node 3 bias

% output layer
b_4_1 = rand(); % node 1 bias
b_4_2 = rand(); % node 2 bias
b_4_3 = rand(); % node 3 bias

% learning rate 
eta = 0.001;

% epochs
epochs = 2;

% backprop
[no_samples,m] = size(IrisDataTrain);
for i = 1:epochs
    for j = 1:no_samples
        % extract parameters
        x_1 = IrisDataTrain(j,1);
        x_2 = IrisDataTrain(j,2);
        x_3 = IrisDataTrain(j,3);
        x_4 = IrisDataTrain(j,4);
        y = IrisDataTrain(j,5);
        % Hidden layer 1
        % linear combination of node v_{layer}_{node} (output of adder)
        v_2_1 = 1*b_2_1 + x_1*w_11(1) + x_2*w_12(1) + x_3*w_13(1) + x_4*w_14(1); % summation node 1
        v_2_2 = 1*b_2_2 + x_1*w_11(2) + x_2*w_12(2) + x_3*w_13(2) + x_4*w_14(2); % summation node 2
        v_2_3 = 1*b_2_3 + x_1*w_11(3) + x_2*w_12(3) + x_3*w_13(3) + x_4*w_14(3); % summation node 3
        v_2_4 = 1*b_2_4 + x_1*w_11(4) + x_2*w_12(4) + x_3*w_13(4) + x_4*w_14(4); % summation node 4
        v_2_5 = 1*b_2_5 + x_1*w_11(5) + x_2*w_12(5) + x_3*w_13(5) + x_4*w_14(5); % summation node 5
        % output of node y_{layer}_{node} (output of activation func)
        y_2_1 = sigmoid(v_2_1); % activation result node 1
        y_2_2 = sigmoid(v_2_2); % activation result node 2
        y_2_3 = sigmoid(v_2_3); % activation result node 3
        y_2_4 = sigmoid(v_2_4); % activation result node 4
        y_2_5 = sigmoid(v_2_5); % activation result node 5
        % Hidden layer 2
        % linear combination of node v_{layer}_{node} (output of adder)
        v_3_1 = 1*b_3_1 + y_2_1*w_21(1) + y_2_2*w_22(1)+ y_2_3*w_23(1) + y_2_4*w_24(1)+ y_2_5*w_25(1); % summation node 1
        v_3_2 = 1*b_3_2 + y_2_1*w_21(2) + y_2_2*w_22(2)+ y_2_3*w_23(2) + y_2_4*w_24(2)+ y_2_5*w_25(2); % summation node 2
        v_3_3 = 1*b_3_3 + y_2_1*w_21(3) + y_2_2*w_22(3)+ y_2_3*w_23(3) + y_2_4*w_24(3)+ y_2_5*w_25(3); % summation node 3
        % output of node y_{layer}_{node} (output of activation func)
        y_3_1 = sigmoid(v_3_1); % activation result node 1
        y_3_2 = sigmoid(v_3_2); % activation result node 2
        y_3_3 = sigmoid(v_3_3); % activation result node 3
        % Output layer
        % linear combination of node v_{layer}_{node} (output of adder)
        v_4_1 = 1*b_4_1 + y_3_1*w_31(1) + y_3_2*w_32(1) + y_3_3*w_33(1); % summation node 1
        v_4_2 = 1*b_4_2 + y_3_1*w_31(2) + y_3_2*w_32(2) + y_3_3*w_33(2); % summation node 2
        v_4_3 = 1*b_4_3 + y_3_1*w_31(3) + y_3_2*w_32(3) + y_3_3*w_33(3); % summation node 3
        % output of node y_{layer}_{node} (output of activation func)
        y_4_1 = sigmoid(v_4_1); % final output node 1
        y_4_2 = sigmoid(v_4_2); % final output node 2
        y_4_3 = sigmoid(v_4_3); % fina  output node 3

        % Update output layer weights
        
    end
end
% define activation functions 
function phi = relu(x)
    if x < 0
        phi =0;
    else
        phi =x;
    end 
end

function phi = leakyRelu(x)
    if x < 0
        phi =0.01*x;
    else
        phi =x;
    end 
end 

function phi =  sigmoid(x)
    phi = 1/(1+exp(-x));
end

