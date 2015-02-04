# mlperceptron
A Multilayer Perceptron implementation in Vala

# Training Program
Included is a sample training program that takes a text file with JSON data and builds one MLP (Multilayer Perceptron) for each training set.

Example:
[
{'name':'AND', 'count_inputs': 2, 'count_neurons': 50, 'layers': [50,1],
 'samples': [
	{'input': [0,0], 'expected': [0]},
	{'input': [0,1], 'expected': [1]},
	{'input': [1,0], 'expected': [1]},
	{'input': [1,1], 'expected': [1], 'skip': true}
]}
]

This will create one MLP that will AND two input values and output a single value.
name: A label for output
count_inputs: The number of inputs for the MLP
count_neurons: The number of neurons on the first layer.  Consequently this is also the number of outputs for that layer.  So if there are no additional layers then this will be the number of outputs.
layers: An array of integers indicating the number of neurons for each additional layer.  The last number is the output layer, so this indicates the number of outputs from the MLP
samples: An array of JSON objects indicating a training sample with an array of inputs (equal to the number of MLP inputs in count_inputs) and expected outputs (equal in number to the number of outputs/neurons in the output layer).  If skip is set to true then this sample will not be trained, but will be tested.


# API
For a complete example see the "generate_twobit_adder" method in mlperceptron.vala

A Perceptron should be generated with the Perceptron.random() constructor method.  This takes a number of inputs, and the number of neurons (consequently indicating the number of outputs for that layer).
	public Perceptron.random(
						int input_count,
						int neuron_count,
						Perceptron? layer_prev=null
						);
						
The layer_prev value is used internally for attaching layers to the network.

To add a new layer use the "add_next_layer(int neuron_count, double[]? weights=null)" method on a previous layer or the root layer. The new layer will automatically be attached to the end of the network.  neuron_count equals the number of neurons in the new layer, and of course indicates the number of outputs for the layer so if this is the last layer then it will indicate the number of outputs for the network.  weights is an optional array indicate the weights for each neuron.  If null then random weights are generated.  If not null, then the length of the array must equal the number of neurons in neuron_count.

Note: The first neuron in a layer is the bias neuron for that layer.  The bias neuron for each layer always receives an input value of 1.

The following will create a new MLP that takes 4 inputs and has 50 neurons on the first layer, 50 on the next (hidden) layer, and 3 on the last (output) layer.  Since the last layer only has 3 neurons, there are 3 values outputted from the network:

	Perceptron p=new Perceptron.random(4, 50);
	p.add_next_layer(50);
	p.add_next_layer(3);
	
After an MLP is generated it can be trained with the "backpropagate" method.  The following attempts to train the MLP with a few samples for a 2 bit adder:
  double learning_rate = 0.001;
	p.backpropagate({0,0, 0,0}, {0,0,0}, learning_rate);	//0+0
	p.backpropagate({0,0, 0,1}, {0,0,1}, learning_rate);	//0+1
	p.backpropagate({0,0, 1,0}, {0,1,0}, learning_rate);	//0+2
	p.backpropagate({0,0, 1,1}, {0,1,1}, learning_rate);	//0+3

Finally after the MLP is trained the "evaluate" method can be used to calculate a value with the network.  This method takes a double array of size input_count and outputs a double array of size equal to the number of neurons in the output layer:
	p.evaluate({0, 0, 0, 0})); //Should output {0,0,0}

