import random
import csv
import math

class Node():
    def __init__(self, name) -> None:
        self.summation = 0
        self.output = 0
        self.bias = 0
        self.outgoing_weights = []
        self.name = name
        self.local_gradient = None

    def __str__(self) -> str:
        str = f'{self.name}, outgoing weights:\n'
        for weight in self.outgoing_weights:
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
        for i, layer in enumerate(self.layers):
            # Every node except those in output layer have outgoing weights.
            if i < self.width-1:
                next_layer_size = self.layers[i+1].size
                for node in layer.nodes:
                    node.outgoing_weights = [random.random() for j in range(next_layer_size)]

    def initialise_biases(self) -> None:
        for i, layer in enumerate(self.layers):
            # Every node except the input layer has a bias.
            if i > 0:
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
        for i, layer in enumerate(self.layers[:-1]): # No need to popogate from output layer
            next_layer = self.layers[i+1]
            # Project values from current layer onto next layer
            for j, to_node in enumerate(next_layer.nodes):
                # Summation step
                calc = f'n_{i+1}_{j} = {to_node.bias}'
                to_node.summation += to_node.bias 
                # bias + w_0x_0 + w_1x_1 + ... 
                for from_node in layer.nodes:
                    calc+=f'+{from_node.summation}*{from_node.outgoing_weights[j]}'
                    to_node.summation += from_node.summation*from_node.outgoing_weights[j]
                calc+=f'= {to_node.summation}'

                # Activation step
                to_node.output = next_layer.activation_func(to_node.summation)

                # Debug                 
                # print(calc)
                # print(f'Activates to {to_node.output}')
    
        # Output results as list
        result = []     
        output_layer = self.layers[-1]
        for node in output_layer.nodes:
            result.append(node.output)
        return result

    def backward_pass(self, real_output, prediction, learning_rate) -> None:
        # Start at output layer, adjust weights

        # Compute gradient for each node in output layer
        output_layer = self.layers[-1]
        for i, node in enumerate(output_layer.nodes):
            d_output = (real_output[i] -prediction[i]) * derivative[output_layer.activation_func](node.summation)
            node.local_gradient = d_output
            node.bias += learning_rate*d_output
            
        # Update weights leading into output layer
        prev_layer = self.layers[-2]
        for i, node in enumerate(prev_layer.nodes):
            for j, weight in enumerate(node.outgoing_weights):
                node.outgoing_weights[j] += learning_rate * output_layer.nodes[j].local_gradient * node.output


        # Treat hidden layers in reverse, excluding input and output layer
        for i in range(self.width-2, 0, -1):
            layer = self.layers[i]
            for node in layer.nodes:
                node.local_gradient = 
                
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

        net.train([dataTrain[1]], 1, 0.01)
