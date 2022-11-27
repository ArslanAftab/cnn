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
        str = f'\n{self.name}\n'
        str += f"Weights: {self.weights}\n"
        str += f"Bias: {self.bias}\n"
        str += f"Summation: {self.summation}\n"
        str += f"Output: {self.output}\n"
        str += f"Delta: {self.delta}"
        return str


class Layer():
    def __init__(self, layer_number, num_nodes, func) -> None:
        self.size = 0
        self.nodes = []
        self.activation_func = func
        self.layer_number = layer_number
        # Create appropriate amount of nodes
        self.create_nodes(num_nodes)

    def create_nodes(self, num_nodes) -> None:
        for i in range(num_nodes):
            node_name = f'{self.layer_number}_n_{i+1}'
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
            layer = Layer(i, layer_size, activation_funcs[i])
            self.layers.append(layer)
            self.width += 1

    def initialise_weights(self) -> None:
        for i, layer in enumerate(self.layers[1:]):
            # Every node except those in input layer has weights.
            prev_layer_size = self.layers[i].size
            for node in layer.nodes:
                node.weights = [random.uniform(-1, 1)
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
                print(f'Input params: {params}')
                # Encode classification
                output = self.categorize_result(output)

                # Normalise parameters
                params = self.normalise_parameters(params)
                print(f'Normalised params: {params}')
                print(f'Expected output: {output}')

                prediction = self.forward_pass(params)
                print(f'Prediction: {prediction}')
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

        # Grab previous layer (n-th hidden layer)
        prev_layer = self.layers[-2]

        # Grab oututs from prev layer
        prev_layer_x_out = []

        for node in prev_layer.nodes:
            # print(node)
            prev_layer_x_out.append(node.output)

        # print('_____________________________________________')
        for j, node in enumerate(output_layer.nodes):
            # print(node)
            # Find delta for each node in output layer
            error = real_output[j] - prediction[j]
            deriv_func = derivative[output_layer.activation_func]
            summation = node.summation
            # print(f'{node.name} gradient = {real_output[j]} - {prediction[j]} * deriv({node.summation})')
            node.delta = self.output_node_delta(error, deriv_func, summation)
            # print(node)

            # Use delta to find weights for each node in output layer
            for i, weight in enumerate(node.weights):
                # print(f'Weight = {weight} + {learning_rate} * {node.delta} * {prev_layer_x_out[i]}')
                node.weights[i] += learning_rate * \
                    node.delta * prev_layer_x_out[i]

            # Use delta to find the bias for each node in output layer
            node.bias += learning_rate * node.delta

            # print(node)

        # Iterate hidden layers in reverse
        for s in range(self.width-2, 0, -1):
            # Grab the layer we want to update delta, weights, bias for
            hidden_layer = self.layers[s]

            # Grab previous layer (s-1 -th hidden layer)
            prev_layer = self.layers[s-1]

            # Grab oututs from prev layer
            prev_layer_x_out = []

            for node in prev_layer.nodes:
                # print(node)
                prev_layer_x_out.append(node.output)

            # Grab the s+1th layer, needed to compute delta
            next_layer = self.layers[s+1]
            # print(f'Fixing s = {s} using layer s+1 = {s+1} deltas')

            for j, node in enumerate(hidden_layer.nodes):
                # Find delta for each node in s-th hidden layer
                # print(f'Delta for  node {j+1}')
                deriv_func = derivative[hidden_layer.activation_func]
                node.delta = self.hidden_node_delta(j, node, next_layer, deriv_func)
            
                # Use delta to find weight for each node in hidden layer
                for i, weight in enumerate(node.weights):
                    # print(f'Weight = {weight} + {learning_rate} * {node.delta} * {prev_layer_x_out[i]}')
                    node.weights[i] += learning_rate * \
                        node.delta * prev_layer_x_out[i]

                # Use delta to find bias for each node in hidden layer
                node.bias += learning_rate * node.delta

    def output_node_delta(self, error, deriv_func, summation):
        return error * deriv_func(summation)

    def hidden_node_delta(self, j, node , next_layer, deriv_func):
        linear_sum = 0
        # print(node)
        for k, next_layer_node in enumerate(next_layer.nodes):
            # print(f'Using: {next_layer_node}')
            node_delta = next_layer_node.delta
            weight = next_layer_node.weights[j]
            # print(f'Summing {node_delta} * {weight} = {node_delta*weight}')
            linear_sum += node_delta * weight
        delta_value = linear_sum * deriv_func(node.output)
        # print(f'Delta = {linear_sum} * deriv({node.output}) = {delta_value}')
        return delta_value

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

    random.seed(10)
    net = Network([4, 5, 3, 3], [None, leakyRelu, leakyRelu, tanh])
    net.initialise_weights()
    net.initialise_biases()
    print(net)
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

        net.train([dataTrain[1]], 1, 0.001)