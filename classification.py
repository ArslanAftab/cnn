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
            str += f'{weight}\n'
        str += f"Bias: {self.bias}"
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
            self.size += 1

    def __str__(self) -> str:
        str = ""
        for node in self.nodes:
            str += f'{node}\n'
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
            self.width += 1

    def initialise_weights(self) -> None:
        for i, layer in enumerate(self.layers[1:]):
            # Every node except those in input layer has weights.
            prev_layer_size = self.layers[i].size
            for node in layer.nodes:
                node.weights = [random.random()
                                for j in range(prev_layer_size)]

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

                # print(f'{params}: {output}')
                # Normalise parameters
                params = self.normalise_parameters(params)
                print(f'{params}: {output}')

                prediction = self.forward_pass(params)
                print(prediction)
                self.backward_pass(output, prediction, learning_rate)

    def normalise_parameters(self, params):
        mean = sum(params)/len(params)
        variance = sum((x-mean) ** 2 for x in params)/len(params)
        std = math.sqrt(variance)
        normalised = []
        for param in params:
            normal = (param-mean)/std
            normalised.append(normal)

        return normalised

    def forward_pass(self, params) -> list:
        input_layer = self.layers[0]

        # Set input data as values for first layer nodes:
        for i, dat in enumerate(params):
            input_layer.nodes[i].summation = dat

        # Start from inputs and propogate values across layers
        # No need to popogate from output layer
        for i, layer in enumerate(self.layers[1:]):
            # Debug
            # print(f'At layer: {i+1}')
            prev_layer = self.layers[i]
            # Project values from prev layer onto current layer
            for node in layer.nodes:
                # Debug
                # print(f'Node {node.name} = {node.bias}', end=" ")
                node.summation += node.bias
                for j, prev_node in enumerate(prev_layer.nodes):
                    # Debug
                    # print(f'+ {prev_node.summation}*{node.weights[j]} ({prev_node.name}*{j}th weight)', end=" ")
                    node.summation += prev_node.summation * node.weights[j]

                # Activation step
                # Debug
                # print(f' {node.summation} Activate to: {layer.activation_func(node.summation)}')
                node.output = layer.activation_func(node.summation)

        # Output results as list
        result = []
        output_layer = self.layers[-1]
        for node in output_layer.nodes:
            result.append(node.output)
        return result

    def backward_pass(self, real_output, prediction, learning_rate) -> None:
        # Grab output layer
        output_layer = self.layers[-1]

        # Find delta's for the output layer
        for node_index, node in enumerate(output_layer.nodes):
            node.delta = self.find_delta_output(
                node, real_output[node_index], prediction[node_index], output_layer.activation_func) # Works

        for layer_index in range(self.width-1, 0, -1): # layer_index = S
            # print(f'Layer: {layer_index}')
            layer = self.layers[layer_index]

            for node_index, node in enumerate(layer.nodes): # node_index = j
                # print(f'Node: {node.name}')
                for weight_index, weight in enumerate(node.weights): # weight_index = i
                    # print(f'Weight: {weight}')
                    # prev layer 
                    prev_layer = self.layers[layer_index-1]
                    # x_out,i
                    prev_layer_output = prev_layer.nodes[weight_index].output
                    # print(f'Weight += learning rate {learning_rate} * node.delta {node.delta} * prev_layer_output {prev_layer_output}')

                    # print(f'X_out prev layer: {prev_output}')
                    # SGD
                    # Update weight
                    node.weights[weight_index] += learning_rate * \
                        node.delta * prev_layer_output
                    # print(f'Weight updated: {weight}')

                    # Update bias
                    node.bias += learning_rate*node.delta

            if layer_index > 1:
                prev_layer = self.layers[layer_index-1]
                for prev_layer_node_index, prev_layer_node in enumerate(prev_layer.nodes):
                    prev_layer_node.delta = self.find_delta_hidden(
                        layer_index, prev_layer_node_index)

    def find_delta_output(self, node, real, pred, activation_func):
        # print(f'Comparing {real} to {pred}')
        error = real - pred
        # print(f'Error amount is {real- pred}')

        d_activation = derivative[activation_func]
        delta = error * d_activation(node.summation)
        # print(f'Calculation: {error}*derivative_activation({node.name}.summation)')
        # print(f'Delta is: {delta}')
        return delta

    def find_delta_hidden(self, layer_index, prev_layer_node_index):
        current_layer = self.layers[layer_index-1]
        current_node = current_layer.nodes[prev_layer_node_index]
        # print(f'Local hidden delta: {layer_index-1},{current_node.name}')
        d_activation = derivative[current_layer.activation_func]
        next_layer = self.layers[layer_index]
        s = 0

        for node in next_layer.nodes:
            weight = node.weights[prev_layer_node_index]
            node_delta = node.delta
            s += weight * node_delta
        # print(f'Sum of delta {s}')
        delta = s * d_activation(current_node.summation)
        return delta

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
            str += f'layer:{i}\n{layer}\n'
        return str


def logistic_sigmoid(v: float) -> float:
    return 1 / (1 + math.exp(-v*0.5))


def tanh(x: float) -> float:
    return math.tanh(x)


def d_logistic_sigmoid(x: float) -> float:
    return (1 - logistic_sigmoid(x)) * logistic_sigmoid(x)


def d_tanh(x: float) -> float:
    return 1.0 - math.tanh(x)**2


def leakyRelu(x: float) -> float:
    if x < 0:
        return 0.01*x
    else:
        return x


def d_leakyRelu(x: float) -> float:
    if x < 0:
        return 0.01
    else:
        return 1


derivative = {
    logistic_sigmoid: d_logistic_sigmoid,
    tanh: d_tanh,
    leakyRelu: d_leakyRelu
}

if __name__ == "__main__":

    net = Network([4, 5, 3, 3], [None, leakyRelu, leakyRelu, logistic_sigmoid])
    net.initialise_weights()
    net.initialise_biases()

    with open('IrisData.txt', mode='r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        irisData = list(csv_reader)

        # Iris types -> numbers
        for row in irisData:
            if row[4] == 'Iris-setosa':
                row[4] = '0'
            elif row[4] == 'Iris-versicolor':
                row[4] = '1'
            else:
                row[4] = '2'

        # Randomize orders
        random.shuffle(irisData)

        # Turn string -> Float
        irisData = [[float(val) for val in row] for row in irisData]

        # Split data, 70% training 30% test
        dataSize = len(irisData)
        trainingDataSize = round(0.7*dataSize)
        dataTrain = irisData[:trainingDataSize]
        dataTest = irisData[trainingDataSize:]

        net.train([dataTrain[1]], 1, 0.1)
