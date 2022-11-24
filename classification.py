import random
import csv
import math

class Node():
    def __init__(self, name) -> None:
        self.summation = 0
        self.output = 0
        self.bias = 0 
        self.weights = []
        self.name = name
        self.delta = 0
        
    def __str__(self) -> str:
        str = f'{self.name},\nweights:\n'
        for weight in self.weights:
            str+=f'{weight}\n'
        str+=f"Bias: {self.bias}"
        return str

class Layer():
    def __init__(self, num_nodes, func) -> None:
        self.size = 0
        self.nodes = []
        self.activation_func = func

        # Create appropriate amount of nodes
        self.create_nodes(num_nodes)

    def create_nodes(self, num_nodes) -> None:
        for i in range(num_nodes):
            node_name = f'n_{i}'
            new_node = Node(node_name)
            self.nodes.append(new_node)
            self.size+=1

    def __str__(self) -> str:
        str = ""
        for node in self.nodes:
            str+= f'{node}\n'
        return str

class Network():
    def __init__(self, layer_sizes, activation_funcs) -> None:
        self.layers = []
        self.width = 0
        self.create_layers(layer_sizes, activation_funcs)
        
    def create_layers(self, layer_sizes, activation_funcs) -> None:
        for i, layer_size in enumerate(layer_sizes):
            layer = Layer(layer_size, activation_funcs[i])
            self.layers.append(layer)
            self.width+=1

    def initialise_weights(self) -> None:
        for i, layer in enumerate(self.layers[1:]):
            # Every node except those in input layer has weights.
            prev_layer_size = self.layers[i].size
            for node in layer.nodes:
                node.weights = [random.random() for j in range(prev_layer_size)]

    def initialise_biases(self) -> None:
        for i, layer in enumerate(self.layers[1:]):
            # Every node except the input layer has a bias.
            for node in layer.nodes:
                node.bias = random.random()

    def train(self, data_set, epochs, learning_rate) -> None:
        for i in range(epochs):
            for data in data_set:
                # Seperate params and output
                params = data[:-1]
                output = data[-1]
                
                # Encode classification 
                output = self.categorize_result(output)

                prediction = self.forward_pass(params)
                self.backward_pass(output, prediction, learning_rate)

    def forward_pass(self, params) -> list:
        input_layer = self.layers[0]

        # Set input data as values for first layer nodes:
        for i, dat in enumerate(params):
            input_layer.nodes[i].summation = dat
        
        # Start from inputs and propogate values across layers
        for i, layer in enumerate(self.layers[1:]): # No need to popogate from output layer
            prev_layer = self.layers[i]
            # Project values from prev layer onto current layer
            for node in layer.nodes:
                node.summation += node.bias
                for j, prev_node in enumerate(prev_layer.nodes):
                    node.summation += prev_node.summation * node.weights[j]
                
                # # Activation step
                node.output = layer.activation_func(node.summation)
        
        # Output results as list
        result = []     
        output_layer = self.layers[-1]
        for node in output_layer.nodes:
            result.append(node.output)
        return result

    def backward_pass(self, real_output, prediction, learning_rate) -> None:
        # Traverse layer by layer in reverse, excluding input layer
        for layer_num in range(self.width-1, 0, -1):
            # Grab current layer and prev layer
            layer = self.layers[layer_num]
            prev_layer = self.layers[layer_num-1]

            # Grab derivative of activation func
            func = layer.activation_func
            d_func = derivative[func]
            
            # Local gradient for output layer is calculated 
            if layer_num == self.width-1:
                for node_num, node in enumerate(layer.nodes):
                    # Set gradient and bias for each node in output layer
                    node.delta = (real_output[node_num] -prediction[node_num]) * d_func(node.summation)
                    node.bias += learning_rate * node.delta
                    # Adjust all incoming weights for node
                    for weight_num, weight in enumerate(node.weights):
                        node.weights[weight_num] += learning_rate * node.delta * prev_layer.nodes[weight_num].output

            # Local gradient for other layers
            else:
                for node_num, node in enumerate(layer.nodes):
                    # Set gradient and bias for each node in hidden layer
                    next_layer = self.layers[layer_num+1]
                    sum = 0
                    for j, next_node in enumerate(next_layer.nodes):
                        sum += next_node.delta * next_node.weights[j]

                    node.delta = sum * d_func(node.summation)
                    node.bias += learning_rate * node.delta

                    # Adjust all incoming weights for node
                    for weight_num, weight in enumerate(node.weights):
                        node.weights[weight_num] += learning_rate * node.delta * prev_layer.nodes[weight_num].output

    def categorize_result(self, result) -> list:
        output_layer = self.layers[-1]
        output = []
        if output_layer.activation_func == logistic_sigmoid:
            # i.e [0.2 0.8 0.2] 
            output = [0.2 for node in output_layer.nodes]
            output[int(result)] = 0.8
        elif output_layer.activation_func == tanh:
            # i.e [-0.6 -0.6 0.6]
            output = [-0.6 for node in output_layer.nodes]
            output[int(result)] = 0.6
        return output

    def __str__(self) -> str:
        str = ""
        for i, layer in enumerate(self.layers):
            str+= f'layer:{i}\n{layer}\n'
        return str

def logistic_sigmoid(x:float) -> float:
  return 1 / (1 + math.exp(-x))

def tanh(x:float) -> float:
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def d_logistic_sigmoid(x:float) -> float:
    return (1 - logistic_sigmoid(x)) * logistic_sigmoid(x)

def d_tanh(x:float) -> float: 
    return  1 - tanh(x)**2

derivative = {
    logistic_sigmoid: d_logistic_sigmoid,
    tanh: d_tanh
}

if __name__ == "__main__":

    net = Network([4,5,3,3], [None,logistic_sigmoid,logistic_sigmoid,logistic_sigmoid])
    net.initialise_weights()
    net.initialise_biases()

    with open('IrisData.txt', mode='r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        irisData = list(csv_reader)
        
        # Iris types -> numbers
        for row in irisData:
            if row[4] == 'Iris-setosa':
                row[4]='0'
            elif row[4]== 'Iris-versicolor':
                row[4]='1'
            else: 
                row[4]='2'   

        # Randomize orders
        random.shuffle(irisData)

        # Turn string -> Float
        irisData = [[float(val) for val in row] for row in irisData]
        
        # Split data, 70% training 30% test
        dataSize = len(irisData)
        trainingDataSize = round(0.7*dataSize)
        dataTrain = irisData[:trainingDataSize]
        dataTest  = irisData[trainingDataSize:]

        net.train([dataTrain[1]], 10, 0.1)
