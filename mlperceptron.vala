/**
 * 2015-02-04
 *
 * mlperceptron.vala - A Multilayer Perceptron implemented in Vala
 *
 * Jason Gaulin
 *
 * Compile with:
 *
 * valac --pkg gee-1.0 --pkg gio-2.0 --pkg json-glib-1.0 -X -lm -X -O3 -X -march=native mlperceptron.vala
 *
 */
using Gee;

public class Neuron : GLib.Object {
	//public uint index {get; private set;}

	public float[] weights {get; private set;}
	private weak Perceptron layer;

    public float? last_guessed;
    public float? error;
    public bool dropout;


	public Neuron(uint index, float[] weights, Perceptron layer) {
		//this.index=index;
		this.weights=weights;
		this.layer=layer;
/*
		for(int i=0;i<weights.length;i++) {
			if(weights[i]<-1||weights[i]>1) {
				stdout.printf("FAIL.Neuron[%u].%d: %f\n", index, i, weights[i]);
			}
		}*/

		last_guessed=null;
		error=null;
		dropout=false;
	}
	//f = 1/(1+exp(-x))
	//df = f * (1 - f)
	private static float tanh(float input) {
		return Math.tanhf(input);
	}
	private static float dxtanh(float input) {
		return 1.0f-Math.powf(Math.tanhf(input), 2.0f);
	}

    private static float sigmoid(float input) {
		//return 1/(1+Math.pow(Math.E, -input));
		return 1/(1+Math.expf(-input));
    }
    public static float dxsigmoid(float input) {
        //e^x/(1+e^x)^2
        float ex=Math.expf(input);//Math.pow(Math.E, input);
        return (ex/Math.powf(1+ex, 2));
        //float f=sigmoid(input);
        //return f*(1-f);
    }
    /*
    public bool last_guessedb {
    	get {
    		if(this.last_guessed<0)
    			return true;
    		return false;
    	}
    }
    private static float activate(float input) {
    	return tanh(input);
    }

    private static float dxactivate(float input) {
    	return dxtanh(input);
    }
    public static bool to_output(float output) {
    	if(output<0.0f)
    		return false;
    	return true;
    }*/
    
    public bool last_guessedb {
    	get {
    		if(this.last_guessed<0.5f)
    			return true;
    		return false;
    	}
    }
    private static float activate(float input) {
    	return sigmoid(input);
    }

    private static float dxactivate(float input) {
    	return dxsigmoid(input);
    }
    public static bool to_output(float output) {
    	if(output<0.5f)
    		return false;
    	return true;
    }
    public float evaluate(float[] inputs)
        requires (inputs.length==weights.length-1) {
		dropout=false;

        float result=0;
        //First calculate the bias
        result+=1;//*weights[0];

        //Now continue with inputs
        for(int i=1;i<weights.length;i++) {
            result+=inputs[i-1]*weights[i];
        }

        //last_guessed = Math.roundf(activate(result));
        last_guessed = (activate(result));

        /*if(layer.layer_next==null) {
	        if(last_guessed<0)
	        	last_guessed=0;
	        else
	        	last_guessed=1;
	        	}*/
        /*
        if(last_guessed<-1.0f||last_guessed>1.0f) {
        	stdout.printf("FAIL.evaluate: %f\n", last_guessed);
        }*/
        //last_guessed = sigmoid(result);

        return last_guessed;
    }

    public float calculate_error(float expected) 
    requires(last_guessed!=null) {
    	if(layer.layer_next==null) {
    		float lg=(float)to_output(last_guessed);

    		this.error=(expected-lg);
    		//stdout.printf("%f %f - %f\n", error, expected, last_guessed);
    	} else 
			this.error=(expected-last_guessed);
/*
		if(error<-1.0f||error>1.0f||expected<-1.0f||expected>1.0f) {
			stdout.printf("ERR.cerr[]: %f %f %f\n", 
				error, 
				expected,
				last_guessed);
		}*/
		//if(error<0)
		//	error=-(float)error;

		return (float)this.error;
    }
    public float backpropagate(float err)
    requires(last_guessed!=null) {
    	float di=0;
/*
    	if(layer.layer_next==null) {
    		if(last_guessed<=0)
    			di=dxactivate(-1.0f);
    		else
    			di=dxactivate(1.0f);
    	} else*/
    		di=dxactivate(last_guessed);

		/*if(err>1||di>1||last_guessed>1) {
			stdout.printf("ERR.b[%u]: %f %f %f\n", 
				index,
				err, 
				di,
				last_guessed);
		}*/
    	this.error=err*di;
/*
		if(error<-1.0f||error>1.0f||err<-1.0f||err>1.0f) {
			stdout.printf("ERR.backprop[]: %f %f %f\n", 
				error, 
				err,
				last_guessed);
		}*/
    	return error;
    }
	public float backpropagate_update_weights(
		float[] inputs, 
		float learning_rate) 
	requires(error!=null)
	requires(inputs.length==weights.length-1) {
		bool err=false;
		/*if(Math.fabs(weights[0])>1||learning_rate>1||Math.fabs(error)>1) {
			stdout.printf("ERR.buw1[%u]: %f %f %f\n", 
				index,
				weights[0], 
				learning_rate, 
				error);
			err=true;
		}*/

		//First update the bias
		weights[0]=weights[0]+learning_rate*error; //*1
		/*if(weights[0]<-1)
			weights[0]=-1;
		if(weights[0]>1)
			weights[0]=1;*/

		for(int i=1;i<weights.length;i++) {
			float change=learning_rate*error*inputs[i-1];
/*
			if(Math.fabs(weights[i]+change)>1) {
				stdout.printf("ERR.buw2[]%d: W[%f => %f] C[%f]=\t(%f*%f*%f) %f\n", 
					
					i,
					weights[i],
					weights[i]+change, 
					change, 
					learning_rate,
					error,
					inputs[i-1],
					last_guessed);
			}*/
			weights[i]=weights[i]+change;
/*
			if(weights[i]<-1)
				weights[i]=-1;
			if(weights[i]>1)
				weights[i]=1;*/

		}
		/*
		if(Math.fabs(weights[0])>1||learning_rate>1||Math.fabs(error)>1) {
			stdout.printf("ERR.buw3[%u]: %f %f %f\n", 
				index,
				weights[0], 
				learning_rate, 
				error);
			err=true;
		}*/

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
	public static Perceptron load(string filename) {
		Perceptron perceptron=null;;

		var network_file = File.new_for_path (filename);

		if (network_file.query_exists ()) {
		    try {
		        // Open file for reading and wrap returned FileInputStream into a
		        // DataInputStream, so we can read line by line
		        var dis = new DataInputStream (network_file.read ());

		        int version = dis.read_line(null,null).to_int();
		        int size = dis.read_line(null,null).to_int();
		        stdout.printf("NETWORKSIZE:%d\n", size);
		        float[] network = new float[(int)size];

		        for(int d=0;d<network.length;d++) {
		        	network[d]=(float)dis.read_line(null,null).to_double();
					//network[d]=(float)dis.read_uint64();
		        }

				perceptron=Perceptron.import(network);

		    } catch (Error e) {
		        error ("%s", e.message);
		    }
		}
		return perceptron;
	}
	public void save(string filename) {
		stdout.printf("Save DISABLED!!!\n");
		return;
		int version=0;
		//var backup_file = File.new_for_path (filename+".bak");
		var old_file = File.new_for_path (filename);
		var network_file = File.new_for_path (filename+".new");
		//stdout.printf("Save to %s\n", filename);
		float[] network = export();

		//Save generated network to file
		
        try {
        	//Remove any partial file
          network_file.delete(null);
        } catch (Error e) {}
        
        try {/*
        	FileStream fs=FileStream.open(filename+".new", "w");
	        fs.printf("%d%d", version, network.length);

			foreach(float x in network ) {
				//data_stream.put_string("%f\n".printf(x), null);
				fs.printf("%f",x);
			}
			network_file.move(old_file, FileCopyFlags.BACKUP|FileCopyFlags.OVERWRITE);
			*/
        	//stdout.printf("Create File %s\n", network_file.get_uri());
			var file_stream = network_file.create (
				FileCreateFlags.REPLACE_DESTINATION, 
				null);
			//stdout.printf(" CREATED\n");
			DataOutputStream data_stream = new DataOutputStream (file_stream);

			data_stream.put_string("%d\n".printf(version));

			data_stream.put_string("%d\n".printf(network.length));
		    //stdout.printf("NETWORKSIZE:%d\n", network.length);

			foreach(float x in network ) {
				data_stream.put_string("%f\n".printf(x), null);
			}
			data_stream.put_byte ('\0');
			network_file.move(old_file, FileCopyFlags.BACKUP|FileCopyFlags.OVERWRITE);
			
	        try {
	        	//Remove any backup file
	          File.new_for_path (filename+"~").delete(null);
	        } catch (Error e) {}
        } catch (Error e) {
          stdout.printf("Save Error: %s\n", e.message);
        }
	}
	public float[] export() {
		float[] values=new float[2+get_weight_count()];

		int offset=0;
		/*int version=0;
		if(this.layer_prev==null) {
			//Header for input layer only
			values[offset++]=version;
		}*/

		//Number of inputs for this layer (-1 bias)
		values[offset++]=(neurons[0].weights.length-1);
		//Number of neurons/outputs for this layer
		values[offset++]=(neurons.length);
		//stdout.printf("EXPORT [inputs=%d] [neurons %d]\n",(neurons[0].weights.length-1), neurons.length);
		for(int n=0;n<neurons.length;n++) {
			for(int i=0;i<neurons[n].weights.length;i++) {
				values[offset++]=(neurons[n].weights[i]);
			}
		}
		if(layer_next!=null) {
			float[] next=layer_next.export();
			//merge
			offset=0;
			float[] merged=new float[values.length+next.length];
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
	public static Perceptron import(float[] values, 
									Perceptron? layer_prev=null) {
		int offset=0;
		int version=0;
		/*if(layer_prev==null) {
			//Header for input layer only
			version=(int)values[offset++];
			stdout.printf("Import Version[%d]\n", version);
		} else {
			version = 0;//layer_prev.version;
		}*/


		int input_count=(int)values[offset++];
		int neuron_count=(int)values[offset++];
		int weights_for_neurons=neuron_count*(input_count+1); //One more for bias

		stdout.printf("IMPORT [inputs=%d] [neurons=%d]", input_count, neuron_count);
		float[] weights=values[offset:offset+weights_for_neurons];

		offset+=weights_for_neurons;

		Perceptron p=new Perceptron(input_count, neuron_count, weights, layer_prev);

		//stdout.printf("offset[%d] < values.length[%d]?\n", offset, values.length);
		//If still more values, add another layer
		if(offset<values.length) {
			p.layer_next=Perceptron.import(values[offset:values.length], p);
		}

		return p;
	}

	public static float[] GenerateWeights(int input_count, int neuron_count) 
	requires(neuron_count>0)
	requires(input_count>0) {
		float[] results=new float[neuron_count*(input_count+1)];

	    for(int i=0;i<results.length;i++) {
	        results[i]=(float)Random.next_double()*2.0f-1.0f;
	        /*if(results[i]>1||results[i]<-1) {
	        	stdout.printf("FAIL.GenerateWeights[%d]: %f\n", i, results[i]);
	        }*/
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
		float[] weights=GenerateWeights(input_count, neuron_count);

		this(input_count, neuron_count, weights, layer_prev);
	}
	
	public Perceptron(	int input_count,
						int neuron_count,
						float[] weights, 
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
										float[]? weights=null) {

		if(layer_next!=null) {
			return layer_next.add_next_layer(neuron_count, weights);
		}

		float[] w=weights;
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
	* Runs through the evaluation and compares the output with expected
	*/
	public float evaluate_error(float[] inputs, float[] expected) {
		float[] results=evaluate(inputs);

		float error=0;
		for(int i=0;i<expected.length;i++) {
			//float e=expected[i]-results[i];
			if(results[i]!=expected[i])
				error++;
		}
		error/=expected.length;

		return error;
	}
	/**
	 * Evaluate the inputs and return the result.
	 * If this is *not* the output layer (this layer has a next)
	 * then passes the output to the next layer, returning its 
	 * output
	 */
    public float[] evaluate(float[] inputs, float dropout_rate=0.0f) {
    	float[] results=new float[this.neurons.length];

    	int thread_limit=20;

		//Thread<void*>[] threads=new Thread<void*>[thread_limit+1];
		//int threadi=0;
		float layer_dropout_rate = 0.0f;

		//If this is a hidden layer, then dropout a percentage of neurons
		//(never drop input or output neurons)
		if(this.layer_prev!=null&&this.layer_next!=null) {
			layer_dropout_rate = dropout_rate;
		}

    	for(int i=0;i<neurons.length;i++) {
    		//If dropout is enabled for this layer, then 
    		//randomly drop the neuron from this iteration
    		if(dropout_rate>0) {
    			if((float)Random.next_double()<=layer_dropout_rate) {
    				neurons[i].dropout=true;
    				results[i]=0;
    				continue;
    			}
    		}
    		//Do not drop out this neuron
    		neurons[i].dropout=false;
    		results[i]=neurons[i].evaluate(inputs);
    		/*int tid=i;

			threads[threadi] = new Thread<void*>(
				"neuron_%d".printf(tid),
				()=>{
					//stdout.printf("Thread %d\n", i);
					//stdout.printf("%d ",tid);
					stdout.printf(".");
					stdout.flush();
					results[tid]=neurons[tid].evaluate(inputs);
					return null;
				});
			threadi+=1;
			if(threadi>=thread_limit) {
				//Only run 10 at a time
		    	stdout.printf("EndThreads[%d:%d]---", tid, neurons.length);
		    	for(int c=0;c<threadi;c++) {
		    		threads[c].join();
		    	}
		    	stdout.printf("Done\n");
		    	threadi=0;
			}*/
    	}
    	/*
    	//Wait for any remaining threads
    	for(int i=0;i<threadi;i++) {
    		threads[i].join();
    	}*/
    	if(layer_next!=null) {
    		return layer_next.evaluate(results, dropout_rate);
		} else {
			//This is the output layer
			for(int i =0;i<results.length;i++) {
				//results[i]=Math.roundf(results[i]);
				results[i]=(float)Neuron.to_output(results[i]);

			}
			return results;
		}
    }
    public uint evaluate_to_uint(float[] inputs) {
    	float[] results=evaluate(inputs);

    	return Perceptron.boolsd_to_uint(results);
    }
    /**
    * Calculate the error for this layer applicable to the 
    * specified neuron on the previous layer
    * En=Sum(Win*Ei)
    */
	public float get_error_for(int previous_layer_neuron_index) {
		float output=0;
		//Step through each neuron on this layer
		for(int i=0;i<this.neurons.length;i++) {
			//index+1 to account for bias (weight[0] is a bias, not fed by neuron)
			if(neurons[i].dropout==false)
				output+=(neurons[i].weights[previous_layer_neuron_index+1]
						*neurons[i].error);
		}
		//this will be multiplied by f'(N)
		return output;
	}
	public float backpropagate(float[]? inputs, float[] expected, float learning_rate) {
		int64 time_start = new DateTime.now_local ().to_unix ();
		int64 time_eval = 0;
		if(this.layer_prev==null) {
			//Input layer

			//print_arr("Training Inputs: ", inputs);
			//print_arr("Expected Outputs:", expected);

			//stdout.printf("Backprop INPUT\n");

			//Input layer
			//Dropout 50% of the hidden neurons
			evaluate(inputs);//, 0.50f);
			time_eval = new DateTime.now_local ().to_unix ();
			//stdout.printf("backpropagate.evaluate: %d s\n", (int)(time_eval-time_start));
		}
		float e=0;

		if(this.layer_next!=null) {
			//Not output layer

			//stdout.printf("Backprop INPUT OR HIDDEN\n");
			this.layer_next.backpropagate(null, expected, learning_rate);

			for(int i=0;i<neurons.length;i+=1) {
				if(neurons[i].dropout)
					continue;
				float err=layer_next.get_error_for(i);
				e=neurons[i].backpropagate(err);
			}
/*
			int thread_limit=40;
			int ops_per_thread=5;
			int threadi=0;
			if(neurons.length<ops_per_thread)
				ops_per_thread=1;
				ops_per_thread=1;
			//Forward layers are now initialized, so continue with this one
			Thread<void*>[] threads=new Thread<void*>[thread_limit+1];
			for(int i=0;i<threads.length;i++) {
				threads[i]=null;
			}
				
			for(int i=0;i<neurons.length;i+=ops_per_thread) {
				if(neurons[i].dropout)
					continue;
				int tid=i;
				int currentthreadid=threadi;
				//If the thread already exists, wait for it to finish
				if(threads[threadi]!=null) {
					//stdout.printf("StopThread %d\n", currentthreadid);
					threads[threadi].join();
					threads[threadi]=null;
				}

				threads[threadi] = new Thread<void*>(
					"neuron_%d".printf(tid),
					()=>{
						//stdout.printf("Thread %d\n", currentthreadid);
						float error=0;
						for(int c=0;c<ops_per_thread;c++) {
							if(neurons.length<=tid+c)
								break;
							layer_next.get_error_for(tid+c);
							neurons[tid+c].backpropagate(error);
						}
						
						return null;
					});
				threadi++;

				//Reset to beginning of thread list
				if(threadi>=threads.length-1) {
					threadi=0;
				}

				 / *
					for(int c=0;c<threadi;c++) {
						threads[c].join();
					}
					threadi=0;
				}* /

			}
			
			for(int i=0;i<threads.length;i++) {
				if(threads[i]!=null)
					threads[i].join();
			}*/
		} else {
			//stdout.printf("Backprop OUTPUT\n");
			//Output layer
			//NOTE: No point in splitting this into threads since
			//it's a simple, limited operation.  Thread overhead
			//with this would kill performance.
			float total_errors=0;
			for(int i=0;i<neurons.length;i++) {
				e=neurons[i].calculate_error(expected[i]);
				neurons[i].backpropagate(e);
        		float lg=(float)neurons[i].last_guessedb;
        		if(lg!=expected[i])
        			total_errors++;
			}
			//Adjust learning_rate
			//stdout.printf("LR %f =>", learning_rate);
			learning_rate*=(1-total_errors/neurons.length);
			if(learning_rate>1.0f)
				learning_rate=1.0f;
			else if(learning_rate<=0.0001f)
				learning_rate=0.0001f;
			//stdout.printf("LR %f\n", learning_rate);
		}
		//Now update the weights
		if(this.layer_prev==null) {
			//Based on the output layer, adjust the learning_rate
			backpropagate_update_weights(inputs, learning_rate);
			int64 time_uw = new DateTime.now_local ().to_unix ();
			//stdout.printf("backpropagate.update_weights: %d s\n", (int)(time_uw-time_eval));
		}

		//Return error from output layer
		return e;
	}
	public void backpropagate_update_weights(float[] inputs, float learning_rate) {
		float[] results=new float[neurons.length];

		//stdout.printf("Layer %u\n", index);
		for(int i=0;i<neurons.length;i++) {
			if(neurons[i].dropout==false)
				results[i]=neurons[i].backpropagate_update_weights(inputs, learning_rate);
			else
				results[i]=0;
		}


		if(this.layer_next!=null) {
			this.layer_next.backpropagate_update_weights(results, learning_rate);
		}
	}
	public static float[] uint_to_boolsd(uint value, int array_size) {
	        //Explode uint to bool array
        float[] outputs=new float[array_size];
        for(int i=0;i<outputs.length;i++) {
            outputs[i]=(float)((bool)(value>>i)&1);
        }

        return outputs;
    }
    public static uint boolsd_to_uint(float[] bools) {
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
/*
void print_arr (string label, float[] a) {
	stdout.printf("%s\n", label);

	for(int i=0;i<a.length;i++) {
		stdout.printf("%d: %f\n", i, a[i]);
	}
	stdout.printf("\n");
}*/
void print_arr (string label, float[] a) {
	stdout.printf("%s\t", label);

	for(int i=0;i<a.length;i++) {
		//stdout.printf("%d: %f\n", i, a[i]);
		stdout.printf("%f ", a[i]);
	}
	stdout.printf("\n");
}

public class TrainingSet {
	public class Sample {
		public float[] inputs;
		public float[] expected;

		public bool skip;

		public Sample(float[] inputs, float[] expected, bool skip) {
			this.inputs = inputs;
			this.expected = expected;
			this.skip = skip;
		}
	}

	public string name;
	public string filename;

	public Sample[] samples;

	public Perceptron perceptron;

	public TrainingSet.load(string filename, string name) {
		this.filename = filename;
		this.name = name;

		perceptron = Perceptron.load(filename);

	}
	public TrainingSet.generate(string filename, 
						string name, int count_inputs, 
						int count_neurons, int[] layer_neurons) {
						//float learning_rate, int training_iterations) {
		this.name = name;
		this.filename = filename;

		perceptron=new Perceptron.random(
			count_inputs, count_neurons);
		//foreach(Json.node neuron_node in obj.get_array_member("layers").get_elements()) {
		for(int i=0;i<layer_neurons.length;i++) {
			perceptron.add_next_layer(layer_neurons[i]);
		}

	}
	public void save() {
		perceptron.save(filename);
	}

	public void test(string label, bool full_details) {

		stdout.printf("Testing [%s] %s...\n", name, label);
		float avg=0;
		for(int i=0;i<samples.length;i++) {
			string l="";
			if(full_details)
				l="%s %s[Sample:%d] %s".printf(label, name, i, samples[i].skip?"Not Trained":"Trained");
			else
				stdout.printf("%d ", i);
			float[] results=perceptron.evaluate(samples[i].inputs);
			float accuracy=print_results(
				l, 
				samples[i].inputs, results, samples[i].expected, full_details);
			avg+=accuracy;
		}
		avg/=samples.length;
		stdout.printf("\nACCURACY: %f%%\n", avg*100);
	}
	public void run(float learning_rate, int training_iterations) {
		int opct=(int)Math.floor(training_iterations*0.01);
		//int fpct=1;//(int)Math.floor(training_iterations*0.10);
		if(opct==0) {
			opct=1;
		}
		//fpct=opct*50;

		stdout.printf("Training [%s], learning_rate=%f, for %d generations (1%% = %d generations)...\n", name, learning_rate, training_iterations,opct);

		for(int g=0;g<training_iterations;g++) {
			if(g%opct==0) {
				stdout.printf(".");
				stdout.flush();
				if(g%(opct*25)==0) {
					stdout.printf("\n");
					test("", false);
					save();
				}
			}
			//Loop over all training samples
			for(int i=0;i<samples.length;i++) {
				if(samples[i].skip)
					continue;
				//stdout.printf(".");
				//stdout.flush();
				perceptron.backpropagate(
					samples[i].inputs, 
					samples[i].expected, 
					learning_rate);
			}
			if(g==training_iterations-1) {
				stdout.printf("*\n");
			}
		}
		save();
		test("PostTraining", true);
		stdout.printf("************************************\n************************************\n************************************\n\n");
	}
}

	float learning_rate = 0.001f;
	int iterations=10000;



/**
 * An example function which creates a Perceptron trained to add two
 * two bit numbers (inputs accepted as 4 bits)
 */
 /*
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
}*/
public float print_results(string message, 
	float[] inputs, float[] results, float[] expected, 
	bool full_details=false) {
	if(message!="")
		stdout.printf("%s\n", message);
	if(full_details&&inputs.length<20&&expected.length<20) {
		print_arr(" INPUT:", inputs);
		print_arr(" OUTPUT:", results);
		print_arr(" EXPECTED:", expected);
	}
	bool match=true;
	float correct=0;
	for(int i=0;i<expected.length;i++) {
		if(results[i]!=expected[i])
			match=false;
		else
			correct++;
	}
	float error = (correct/expected.length);
	//stdout.printf("RESULTS :: %s\n", match==true?"SUCCESS":"***FAIL***");
	stdout.printf("RESULTS :: Correct: %f%% %s\n", error*100, (error>=1)?"*****":"-----");

	return error;
}
