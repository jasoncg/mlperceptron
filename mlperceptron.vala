/**
 * 2015-02-04
 *
 * mlperceptron.vala - A Multilayer Perceptron implemented in Vala
 *
 * Jason Gaulin
 *
 * Compile with:
 *
 * valac --pkg gee-1.0 --pkg gio-2.0 --pkg json-glib-1.0 -X -lm mlperceptron.vala
 *
 */
using Gee;

public class Neuron : GLib.Object {
	public uint index {get; private set;}

	public double[] weights {get; private set;}
	private weak Perceptron layer;

    public double? last_guessed;
    public double? error;

	public Neuron(uint index, double[] weights, Perceptron layer) {
		this.index=index;
		this.weights=weights;
		this.layer=layer;

		last_guessed=null;
		error=null;
	}
    private static double sigmoid(double input) {
		return 1/(1+Math.pow(Math.E, -input));
    }
    public static double disigmoid(double input) {
        //e^x/(1+e^x)^2
        double ex=Math.pow(Math.E, input);
        return (ex/Math.pow(1+ex, 2));
    }

    public double evaluate(double[] inputs)
        requires (inputs.length==weights.length-1) {

        double result=0;
        //First calculate the bias
        result+=1*weights[0];
        //Now continue with inputs
        for(int i=1;i<weights.length;i++) {
            result+=inputs[i-1]*weights[i];
        }
        last_guessed = Math.round(sigmoid(result));
        //last_guessed = sigmoid(result);

        return last_guessed;
    }

    public double calculate_error(double expected) 
    requires(last_guessed!=null) {
		this.error=(expected-last_guessed);
		//if(error<0)
		//	error=-(double)error;

		return (double)this.error;
    }
    public double backpropagate(double err)
    requires(last_guessed!=null) {
    	this.error=err*disigmoid(last_guessed);

    	return error;
    }
	public double backpropagate_update_weights(
		double[] inputs, 
		double learning_rate) 
	requires(error!=null)
	requires(inputs.length==weights.length-1) {
		//First update the bias
		weights[0]=weights[0]+learning_rate*error;

		for(int i=1;i<weights.length;i++) {

			weights[i]=weights[i]+learning_rate*error*inputs[i-1];
		}

		return evaluate(inputs);
	}
}

public class Perceptron : GLib.Object {
	public uint index {get; private set;}
	public Neuron[] neurons {get; private set;}
	public Perceptron? layer_prev;
	public Perceptron? layer_next;

	public uint get_weight_count() {
		uint c=0;

		for(int n=0;n<neurons.length;n++) {
			c+=neurons[n].weights.length;
		}

		return c;
	}
	public double[] export() {
		double[] values=new double[2+get_weight_count()];

		int offset=0;
		//Number of inputs for this layer (-1 bias)
		values[offset++]=(neurons[0].weights.length-1);
		//Number of neurons/outputs for this layer
		values[offset++]=(neurons.length);
		
		for(int n=0;n<neurons.length;n++) {
			for(int i=0;i<neurons[n].weights.length;i++) {
				values[offset++]=(neurons[n].weights[i]);
			}
		}
		if(layer_next!=null) {
			double[] next=layer_next.export();
			//merge
			offset=0;
			double[] merged=new double[values.length+next.length];
			for(int i=0;i<values.length;i++) {
				merged[offset++]=values[i];
			}
			for(int i=0;i<next.length;i++) {
				merged[offset++]=next[i];
			}
			values=merged;
		}
		return values;
	}
	/**
	 * Generate a Perceptron with the specified configuration
	 */
	public static Perceptron import(double[] values, 
									Perceptron? layer_prev=null) {
		int offset=0;
		int input_count=(int)values[offset++];
		int neuron_count=(int)values[offset++];
		int weights_for_neurons=neuron_count*(input_count+1); //One more for bias

		stdout.printf("IMPORT [inputs=%d] [neurons=%d]", input_count, neuron_count);
		double[] weights=values[offset:offset+weights_for_neurons];

		offset+=weights_for_neurons;

		Perceptron p=new Perceptron(input_count, neuron_count, weights, layer_prev);

		stdout.printf("offset[%d] < values.length[%d]?\n", offset, values.length);
		//If still more values, add another layer
		if(offset<values.length) {
			p.layer_next=Perceptron.import(values[offset:values.length], p);
		}

		return p;
	}

	public static double[] GenerateWeights(int input_count, int neuron_count) 
	requires(neuron_count>0)
	requires(input_count>0) {
		double[] results=new double[neuron_count*(input_count+1)];

	    for(int i=0;i<results.length;i++) {
	        results[i]=Random.next_double()*2.0f-1.0f;
	    }
	    return results;
	}

    public string to_string() {
        string output="PERCEPTRON{%u}(neurons:%d)".printf(index, neurons.length);
        if(layer_next!=null) {
        	if(layer_prev==null) {
        		output+="INPUT LAYER\n";
        	} else
	        	output+="HIDDEN LAYER\n";
	        output+="%s".printf(layer_next.to_string());
        } else
        output+="OUTPUT LAYER\n";
/*        for(int i=0;i<layers.length;i++) {
            output+="%s\n".printf(layers[i].to_string());
        }
        output+=")\n";*/
        return output;
    }

	public Perceptron.random(
						int input_count,
						int neuron_count,
						Perceptron? layer_prev=null
						) {
		double[] weights=GenerateWeights(input_count, neuron_count);

		this(input_count, neuron_count, weights, layer_prev);
	}
	
	public Perceptron(	int input_count,
						int neuron_count,
						double[] weights, 
						Perceptron? layer_prev=null
						) 
	requires(weights.length==neuron_count*(input_count+1)) {
		this.layer_prev=layer_prev;
		this.layer_next=null;

		if(layer_prev==null)
			index=0;
		else
			index=layer_prev.index+1;

		int weight_index=0;
		int weights_per_neuron=input_count+1;

		this.neurons=new Neuron[neuron_count];
		//Create one input for each neuron in the previous layer
		for(int i=0;i<neurons.length;i++) {
			Neuron n=new Neuron(i, 
								weights[weight_index:weight_index+weights_per_neuron],
								this);
			weight_index+=weights_per_neuron;
			neurons[i]=n;
		}
	}

	/**
	 * Add another layer after this one.
	 * Note that neuron_count represents not only the number of neurons in
	 * the next layer, but also the number of *outputs* for that layer.
	 */
	public Perceptron add_next_layer(	int neuron_count, 
										double[]? weights=null) {

		if(layer_next!=null) {
			return layer_next.add_next_layer(neuron_count, weights);
		}

		double[] w=weights;
		if(w==null) {
			w = GenerateWeights(this.neurons.length, neuron_count);
		}
		Perceptron next=new Perceptron(
			this.neurons.length,
			neuron_count, 
			w, 
			this);
		layer_next=next;

		return layer_next;
	}
	/**
	 * Evaluate the inputs and return the result.
	 * If this is *not* the output layer (this layer has a next)
	 * then passes the output to the next layer
	 */
    public double[] evaluate(double[] inputs) {
    	double[] results=new double[this.neurons.length];
    	for(int i=0;i<neurons.length;i++) {
    		results[i]=neurons[i].evaluate(inputs);
    	}
    	if(layer_next!=null) {
    		return layer_next.evaluate(results);
		} else {
			//This is the output layer
			return results;
		}
    }
    public uint evaluate_to_uint(double[] inputs) {
    	double[] results=evaluate(inputs);

    	return Perceptron.boolsd_to_uint(results);
    }

	public double get_error_for(int previous_layer_neuron_index) {
		double output=0;
		for(int i=0;i<this.neurons.length;i++) {
			output+=neurons[i].weights[previous_layer_neuron_index+1]*neurons[i].error;
		}

		return output;
	}
	public void backpropagate(double[]? inputs, double[] expected, double learning_rate) {

		if(this.layer_prev==null) {
			//print_arr("Training Inputs: ", inputs);
			//print_arr("Expected Outputs:", expected);

			//stdout.printf("Backprop INPUT\n");
			//Input layer
			evaluate(inputs);
		}

		if(this.layer_next!=null) {
			//stdout.printf("Backprop INPUT OR HIDDEN\n");
			this.layer_next.backpropagate(null, expected, learning_rate);

			//Forward layers are now initialized, so continue with this one
			for(int i=0;i<neurons.length;i++) {
				double e=layer_next.get_error_for(i);
				neurons[i].backpropagate(e);
			}
		} else {
			//stdout.printf("Backprop OUTPUT\n");
			//Output layer
			for(int i=0;i<neurons.length;i++) {
				double e=neurons[i].calculate_error(expected[i]);
				neurons[i].backpropagate(e);
			}
		}
		//Now update the weights
		if(this.layer_prev==null) {
			backpropagate_update_weights(inputs, learning_rate);
		}
	}
	public void backpropagate_update_weights(double[] inputs, double learning_rate) {
		double[] results=new double[neurons.length];

		for(int i=0;i<neurons.length;i++) {
			results[i]=neurons[i].backpropagate_update_weights(inputs, learning_rate);
		}


		if(this.layer_next!=null) {
			this.layer_next.backpropagate_update_weights(results, learning_rate);
		}
	}
	public static double[] uint_to_boolsd(uint value, int array_size) {
	        //Explode uint to bool array
        double[] outputs=new double[array_size];
        for(int i=0;i<outputs.length;i++) {
            outputs[i]=(double)((bool)(value>>i)&1);
        }

        return outputs;
    }
    public static uint boolsd_to_uint(double[] bools) {
        uint mask=1;
        uint result=0;

        for(uint i=0;i<bools.length;i++) {
            if(bools[i]>0.5) {
                result|=mask;
            }
            mask<<=1;
        }
        
        return result;	
    }
	public static bool[] uint_to_bools(uint value, int array_size) {
	        //Explode uint to bool array
        bool[] outputs=new bool[array_size];
        for(int i=0;i<outputs.length;i++) {
            outputs[i]=(bool)(value>>i)&1;
        }

        return outputs;
    }
    public static uint bools_to_uint(bool[] bools) {
        uint mask=1;
        uint result=0;

        for(uint i=0;i<bools.length;i++) {
            if(bools[i]==true) {
                result|=mask;
            }
            mask<<=1;
        }
        
        return result;	
    }
}
// Another way to print an array
void print_arr (string label, double[] a) {
	stdout.printf("%s\n", label);

	for(int i=0;i<a.length;i++) {
		stdout.printf("%d: %f\n", i, a[i]);
	}
	stdout.printf("\n");
}
	double learning_rate = 0.001;
	int iterations=10000;



/**
 * An example function which creates a Perceptron trained to add two
 * two bit numbers (inputs accepted as 4 bits)
 */
Perceptron generate_twobit_adder() {

	Perceptron p=new Perceptron.random(4, 50);
	p.add_next_layer(50);
	p.add_next_layer(3);

	stdout.printf("PRETRAINING\n");
	for(int l1a=0;l1a<=1;l1a++)
	for(int l1b=0;l1b<=1;l1b++)
	for(int l2a=0;l2a<=1;l2a++)
	for(int l2b=0;l2b<=1;l2b++) {
		print_arr("%d%d %d%d".printf(l1a,l1b, l2a,l2b), 
			p.evaluate({l1a, l1b, l2a,l2b}));
	}
	stdout.printf("Train for %d iterations...\n", iterations);
	for(int i=0;i<iterations;i++) {
		stdout.printf(".");
		stdout.flush();
			//2 bit Adder (outputs 3 bits)
		p.backpropagate({0,0, 0,0}, {0,0,0}, learning_rate);	//0+0
		p.backpropagate({0,0, 0,1}, {0,0,1}, learning_rate);	//0+1
		p.backpropagate({0,0, 1,0}, {0,1,0}, learning_rate);	//0+2
		p.backpropagate({0,0, 1,1}, {0,1,1}, learning_rate);	//0+3

		p.backpropagate({0,1, 0,0}, {0,0,1}, learning_rate);	//1+0
		p.backpropagate({0,1, 0,1}, {0,1,0}, learning_rate);	//1+1
		p.backpropagate({0,1, 1,0}, {0,1,1}, learning_rate);	//1+2
		p.backpropagate({0,1, 1,1}, {1,0,0}, learning_rate);	//1+3

		p.backpropagate({1,0, 0,0}, {0,1,0}, learning_rate);	//2+0
		p.backpropagate({1,0, 0,1}, {0,1,1}, learning_rate);	//2+1
		p.backpropagate({1,0, 1,0}, {1,0,0}, learning_rate);	//2+2
		p.backpropagate({1,0, 1,1}, {1,0,1}, learning_rate);	//2+3

		p.backpropagate({1,1, 0,0}, {0,1,1}, learning_rate);	//3+0
		p.backpropagate({1,1, 0,1}, {1,0,0}, learning_rate);	//3+1
		p.backpropagate({1,1, 1,0}, {1,0,1}, learning_rate);	//3+2
		p.backpropagate({1,1, 1,1}, {1,1,1}, learning_rate);	//3+3
	}

	for(int l1a=0;l1a<=1;l1a++)
	for(int l1b=0;l1b<=1;l1b++)
	for(int l2a=0;l2a<=1;l2a++)
	for(int l2b=0;l2b<=1;l2b++) {
		print_arr("%d%d %d%d".printf(l1a,l1b, l2a,l2b), 
			p.evaluate({l1a, l1b, l2a,l2b}));
	}
	return p;
}
public void print_results(string message, double[] inputs, double[] results, double[] expected) {
	stdout.printf("%s\n", message);
	
	//print_arr("INPUT:", inputs);
	//print_arr("OUTPUT:", results);
	//print_arr("EXPECTED:", expected);
	bool match=true;
	for(int i=0;i<expected.length;i++) {
		if(results[i]!=expected[i])
			match=false;
	}
	stdout.printf("RESULTS :: %s\n", match==true?"SUCCESS":"***FAIL***");
}
public static int main(string[] args) {
	if(args.length!=2) {
		stdout.printf("A training file is required\n");
		return 1;
	}
	stdout.printf("Training File: %s\n", args[1]);

	//Read a training file
	Json.Parser parser;
    try {
    	//Get Perceptron settings
    	parser=new Json.Parser();
    	parser.load_from_file(args[1]);

	} catch(Error e) {
		error("%s", e.message);
		return 1;
	}

	Json.Node node = parser.get_root();

	//Step through each NN perceptron
	foreach(var o in node.get_array().get_elements()) {
		Json.Object obj=o.get_object();
		string name="NOT NAMED";
		int input_count=0;
		int neuron_count=0;
		int training_iterations=100;
		double learning_rate=0.01;

		if(obj.has_member("name"))
			name=obj.get_string_member("name");
		if(obj.has_member("training_iterations"))
			training_iterations=(int)obj.get_int_member("training_iterations");
		if(obj.has_member("learning_rate"))
			learning_rate=obj.get_double_member("learning_rate");
		
		

		input_count=(int)obj.get_int_member("count_inputs");
		neuron_count=(int)obj.get_int_member("count_neurons");
		stdout.printf("Build Perceptron '%s'[inputs=%d input_neurons=%d]...\n", name, input_count, neuron_count);

		Perceptron p=new Perceptron.random(input_count, neuron_count);
		foreach(Json.Node neuron_node in obj.get_array_member("layers").get_elements()) {

			p.add_next_layer((int)neuron_node.get_int());
		}
		//Now train the Perceptron
		stdout.printf("Training...\n");

		int sample_count=0;
		foreach(Json.Node sample_node in obj.get_array_member("samples").get_elements()) {
			Json.Object sample=sample_node.get_object();

			Json.Array inputs_ja=sample.get_array_member("input");
			Json.Array expected_ja=sample.get_array_member("expected");
			
			double[] inputs=new double[inputs_ja.get_length()];
			double[] expected=new double[expected_ja.get_length()];

			for(int i=0;i<inputs.length;i++) {
				inputs[i]=inputs_ja.get_double_element(i);
			}
			for(int i=0;i<expected.length;i++) {
				expected[i]=expected_ja.get_double_element(i);
				//expected[i]=expected_ja.get_boolean_element(i);
			}

			bool skipped=true;
			//If "skip", then don't train with this one
			if(!sample.has_member("skip")) {
				skipped=false;
				for(int iteration=0;iteration<training_iterations;iteration++) {
					p.backpropagate(inputs, expected, learning_rate);
				}
			}

			//Now test
			double[] results=p.evaluate(inputs);
			print_results("PostTraining %s[Sample:%d] %s".printf(name, sample_count, skipped?"Not Trained":"Trained"), 
				inputs, results, expected);
			sample_count++;
		}
	}

	return 0;
}
