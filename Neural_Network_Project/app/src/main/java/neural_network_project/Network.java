package neural_network_project;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import neural_network_project.Helper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;


public class Network{
    int num_layers;
    int[] networkSize;
    List<INDArray> weights = new ArrayList<>();
    List<INDArray> biases = new ArrayList<>();


    public Network(int[] size){
        this.num_layers = size.length;
        this.networkSize = size;
        for(int i=1;i < num_layers;i++){
            // Number of neurons for current layer
            int x = size[i];
            INDArray bias = Nd4j.randn(x, 1);
            this.biases.add(bias);
        }

        for(int i=0; i < num_layers -1; i++){
            // Number of neurons for current layer
            int x = size[i];
            // Number of neurons for next layer
            int y = size[i+1];
            INDArray weight = Nd4j.randn(y,x);
            this.weights.add(weight);
        }
    }


    public INDArray feedforward(INDArray a){
        for(int i = 0; i < this.biases.size(); i++){
            INDArray weightsMatrix = this.weights.get(i);
            INDArray biasesMatrix = this.biases.get(i);
            a = Nd4j.add(weightsMatrix.mmul(a),biasesMatrix);
            a = Transforms.sigmoid(a);
        }

        return a;
    }


    public List<List<INDArray>> backpropagation(INDArray output_activations, INDArray desiredOutput){
        List<INDArray> gradient_biases = new ArrayList<>();
        List<INDArray> gradient_weights = new ArrayList<>();
        
        for(INDArray weight:this.weights){
            gradient_weights.add(Nd4j.zerosLike(weight));
        }
        for(INDArray bias:this.biases){
            gradient_weights.add(Nd4j.zerosLike(bias));
        }

        INDArray activation = output_activations;
        List<INDArray> activations = new ArrayList<>();
        activations.add(output_activations);
        List<INDArray> z_vectors = new ArrayList<>();

        for(int i=0; i<this.biases.size(); i++){
            INDArray weightsMatrix = this.weights.get(i);
            INDArray biasesMatrix = this.biases.get(i);
            INDArray z = Nd4j.add(weightsMatrix.mmul(activation),biasesMatrix);
            z_vectors.add(z);
            activation = Transforms.sigmoid(z);
            activations.add(activation);
        }

        INDArray last_activation_layer = activations.get(activations.size()-1);
        INDArray last_z_vector_layer = z_vectors.get(z_vectors.size()-1);
        // Backward pass
        INDArray delta_vector = this.cost_derivative(last_activation_layer, desiredOutput).mul(Transforms.sigmoidDerivative(last_z_vector_layer));
        gradient_biases.set(gradient_biases.size()-1,delta_vector);
        gradient_weights.set(gradient_weights.size()-1,delta_vector.mmul(activations.get(activations.size()-2).transpose()));


        for(int layer=this.num_layers-2; layer > 0; layer--){
            INDArray z = z_vectors.get(layer);
            INDArray sigmoid_prime = Transforms.sigmoidDerivative(z);
            delta_vector = this.weights.get(layer+1).transpose().mmul(delta_vector).mul(sigmoid_prime);

            gradient_biases.set(layer, delta_vector);
            gradient_weights.set(layer, delta_vector.mmul(activations.get(layer-1).transpose()));
        }

        List<List<INDArray>> gradients = new ArrayList<>();
        gradients.add(gradient_biases);
        gradients.add(gradient_weights);
        return gradients;
    }


    public INDArray cost_derivative(INDArray output_activations, INDArray desiredOutput){
        return Nd4j.sub(output_activations,desiredOutput);
    }


    public void update_mini_batch(List<List<INDArray>> mini_batches, double learning_rate){
        List<INDArray> gradient_biases = new ArrayList<>();
        List<INDArray> gradient_weights = new ArrayList<>();
        
        for(INDArray weight:this.weights){
            gradient_weights.add(Nd4j.zerosLike(weight));
        }
        for(INDArray bias:this.biases){
            gradient_weights.add(Nd4j.zerosLike(bias));
        }


        // Iterate throught the mini batches
        for(List<INDArray> mini_batch : mini_batches){
            INDArray output_activations = mini_batch.get(0);
            INDArray desiredOutput = mini_batch.get(1);

            List<List<INDArray>> gradients = this.backpropagation(output_activations, desiredOutput);
            List<INDArray> delta_gradient_biases = gradients.get(0);
            List<INDArray> delta_gradient_weights = gradients.get(1);

            for(int i=0; i<delta_gradient_biases.size(); i++){
                INDArray gradient_bias = gradient_biases.get(i);
                INDArray delta_gradient_bias = delta_gradient_biases.get(i);
                INDArray new_gradient_bias = Nd4j.add(gradient_bias, delta_gradient_bias);

                gradient_biases.set(i, new_gradient_bias);
            }

            for(int i=0; i<delta_gradient_weights.size(); i++){
                INDArray gradient_weight = gradient_weights.get(i);
                INDArray delta_gradient_weight = delta_gradient_weights.get(i);
                INDArray new_gradient_weight = Nd4j.add(gradient_weight, delta_gradient_weight);

                gradient_weights.set(i, new_gradient_weight);
            }

            // Updating the network weights and biases base on the average gradient of the mini batches
            // Update the weights
            for(int i=0; i<this.weights.size(); i++){
                INDArray current_weight = this.weights.get(i);
                INDarray gradient_weight_sum = gradient_weights.get(i);

                INDArray average_weight_gradient = gradient_weight_sum.muli(learning_rate/mini_batches.size());
                INDArray new_weight = Nd4j.sub(current_weight, average_weight_gradient);

                this.weights.set(i, new_weight);
            }

            // Update the biases
            for(int i=0; i<this.biases.size(); i++){
                INDArray current_bias = this.biases.get(i);
                INDarray gradient_bias_sum = gradient_biases.get(i);

                INDArray average_bias_gradient = gradient_bias_sum.muli(learning_rate/mini_batches.size());
                INDArray new_bias = Nd4j.sub(current_bias, average_bias_gradient);

                this.biases.set(i, new_bias);
            }
        }
    }


    public int evaluate(List<List<INDArray>> test_datas){
        int correct_predictions = 0;
        List<List<Integer>> test_results = new ArrayList<>();

        for(List<INDArray> test_data:test_datas){
            INDArray raw_input = test_data.get(0);
            INDArray desired_output = test_data.get(1);

            INDArray finale_output_layer = this.feedforward(raw_input);
            Integer class_index = Nd4j.argMax(finale_output_layer).getInt(0);

            List<Integer> test_result = new ArrayList<>();
            test_result.add(class_index);
            test_result.add(desired_output);

            test_results.add(test_result);
        }

        for(List<Integer> result:test_results){
            if(result.get(0) == result.get(1)) correct_predictions++;
        }

        return correct_predictions;
    }
}