package neural_network_project;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;

import javax.sound.midi.SysexMessage;

import java.util.Arrays;
import neural_network_project.Helper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
// Java serialization 
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.Serializable;


// This class implements Serializable to save intances of the class
public class Network1 implements Serializable{
    int num_layers;
    int[] networkSize;
    // Weights and biases are store in seperate lists.
    List<INDArray> weights = new ArrayList<>();
    List<INDArray> biases = new ArrayList<>();


    // Constructor takes in an array of int to represent the network's size
    public Network1(int[] size){
        this.num_layers = size.length;
        this.networkSize = size;
        // Initialize the network's biases. Skip the first layer since we don't need a bias vector for that
        for(int i=1;i < num_layers;i++){
            // Number of neurons for current layer
            int x = size[i];
            INDArray bias = Nd4j.randn(x, 1);
            this.biases.add(bias);
        }

        // Same thing with the weights, except the weights are represented as matricies
        for(int i=0; i < num_layers -1; i++){
            // Number of neurons for current layer
            int x = size[i];
            // Number of neurons for next layer
            int y = size[i+1];
            INDArray weight = Nd4j.randn(y,x);
            this.weights.add(weight);
        }
    }


    // Method to feedforward inputs through the whole network
    public INDArray feedforward(INDArray a){
        // Iterate through each layers to compute output
        for(int i = 0; i < this.biases.size(); i++){
            INDArray weightsMatrix = this.weights.get(i);
            INDArray biasesMatrix = this.biases.get(i);
            a = weightsMatrix.mmul(a);
            a = a.add(biasesMatrix);
            // Apply activation function
            a = Transforms.sigmoid(a);
        }
        return a;
    }


    /*Implementation of stochastic Gradient Descent, this is where the network trains.
    All the bulk works is done in the backpropagation() method */ 
    public void stochasticGradientDescent(List<List<INDArray>> training_datas, int epochs, int mini_batch_size, float learning_rate, List<List<INDArray>> test_datas){
        // Size of each datasets. One for training and one for testing.
        int n_test = test_datas.size();
        int n = training_datas.size();

        // Iterate through the epochs
        for(int i=0; i<epochs; i++){
            // Shuffle up the training datas.
            Collections.shuffle(training_datas);
            List<List<List<INDArray>>> mini_batches = new ArrayList<>();

            // divide the training datas into smaller mini batches
            for(int k=0; k<n; k+=mini_batch_size){
                mini_batches.add(training_datas.subList(k, k+mini_batch_size));
            }

            // Train the network using update_mini_batch() (which uses backpropagation())
            for(List<List<INDArray>> mini_batch : mini_batches){
                this.update_mini_batch(mini_batch, learning_rate);
            }

            // Print out the evaluated accuracy on current epoch
            System.out.println(String.format("Epoch %d: %d / %d", i, this.evaluate(test_datas), n_test));
        }

    }


    // This where backpropagation is implemented, the bulk of neural network computations.
    public List<List<INDArray>> backpropagation(INDArray output_activations, INDArray desiredOutput){
        // Create lists to store the gradient vectors and matricies of biases and weights
        List<INDArray> gradient_biases = new ArrayList<>();
        List<INDArray> gradient_weights = new ArrayList<>();
        
        /*Initialize them with zeros, gradient vectors and matrices have to be the same size of the 
        current biases and weights of the network*/ 
        for(INDArray weight:this.weights){
            gradient_weights.add(Nd4j.zerosLike(weight));
        }
        for(INDArray bias:this.biases){
            gradient_biases.add(Nd4j.zerosLike(bias));
        }

        // output_activations is the raw feedforward activated output of the network.
        INDArray activation = output_activations;
        // Create list to store all activated outputs of layers and add the last output
        List<INDArray> activations = new ArrayList<>();
        activations.add(output_activations);
        // z vectors is the output of each layer before being activated
        List<INDArray> z_vectors = new ArrayList<>();

        // Fill up the two recent lists with activated and inactivated outputs of each layer
        for(int i=0; i<this.biases.size(); i++){
            INDArray weightsMatrix = this.weights.get(i);
            INDArray biasesMatrix = this.biases.get(i);
            INDArray z = weightsMatrix.mmul(activation).add(biasesMatrix);
            z_vectors.add(z);
            activation = Transforms.sigmoid(z);
            activations.add(activation);
        }

        // Get the last activated and inactivated outputs (the last layer)
        INDArray last_activation_layer = activations.get(activations.size()-1);
        INDArray last_z_vector_layer = z_vectors.get(z_vectors.size()-1);

        // Backward pass
        /* delta_vector is the derivative of the loss function with respect to  inactivated 
        output (z). Using the chain rule, we can compute this by multiplying cost_derivative
        (derivative of the loss respect to the last activated ouput) by sigmoid derivative of z
        which is the activation function we are using
        */ 
        INDArray delta_vector = this.cost_derivative(last_activation_layer, desiredOutput).mul(Transforms.sigmoidDerivative(last_z_vector_layer));
        // Compute and add the last layer's biases and weights gradient to the list.
        gradient_biases.set(gradient_biases.size()-1,delta_vector);
        gradient_weights.set(gradient_weights.size()-1,delta_vector.mmul(activations.get(activations.size()-2).transpose()));


        // This is where we comput the gradient of the rest parameters of the network
        for(int l=2; l < this.num_layers; l++){
            //We are moving backward throught the network (last to first)
            INDArray z = z_vectors.get(z_vectors.size() - l);
            INDArray sigmoid_prime = Transforms.sigmoidDerivative(z);
            // Compute the delta_vector using the formula.
            delta_vector = this.weights.get(this.weights.size()-l+1).transpose().mmul(delta_vector).mul(sigmoid_prime);

            // Assign the computed biases and weights gradient to the list
            gradient_biases.set(gradient_biases.size()-l, delta_vector);
            gradient_weights.set(gradient_weights.size()-l, delta_vector.mmul(activations.get(activations.size()-l-1).transpose()));
        }

        // We forged the two gradients to a list and return it so other methods can use it.
        List<List<INDArray>> gradients = new ArrayList<>();
        gradients.add(gradient_biases);
        gradients.add(gradient_weights);
        return gradients;
    }


    // This is for computing the cost derivative (loss function with respect to activated output)
    public INDArray cost_derivative(INDArray output_activations, INDArray desiredOutput){
        return output_activations.sub(desiredOutput);
    }


    // Now we have computed the gradients for the network's parameters. We just need to update them.
    public void update_mini_batch(List<List<INDArray>> mini_batches, float learning_rate){
        List<INDArray> gradient_biases = new ArrayList<>();
        List<INDArray> gradient_weights = new ArrayList<>();

        // Creating and initializing the weights and biases gradients
        for(INDArray weight:this.weights){
            gradient_weights.add(Nd4j.zerosLike(weight));
        }
        for(INDArray bias:this.biases){
            gradient_biases.add(Nd4j.zerosLike(bias));
        }


        // Iterate throught the mini batches
        for(List<INDArray> mini_batch : mini_batches){
            // Get the network's output and the correct labeled output
            INDArray output_activations = mini_batch.get(0);
            INDArray desiredOutput = mini_batch.get(1);

            // Feed them to the backpropagation() method
            List<List<INDArray>> gradients = this.backpropagation(output_activations, desiredOutput);
            // Get the returned gradients
            List<INDArray> delta_gradient_biases = gradients.get(0);
            List<INDArray> delta_gradient_weights = gradients.get(1);

            // Add up the gradients of each mini batches
            for(int i=0; i<delta_gradient_biases.size(); i++){
                INDArray gradient_bias = gradient_biases.get(i);
                INDArray delta_gradient_bias = delta_gradient_biases.get(i);
                INDArray new_gradient_bias = gradient_bias.add(delta_gradient_bias);

                gradient_biases.set(i, new_gradient_bias);
            }

            for(int i=0; i<delta_gradient_weights.size(); i++){
                INDArray gradient_weight = gradient_weights.get(i);
                INDArray delta_gradient_weight = delta_gradient_weights.get(i);
                INDArray new_gradient_weight = gradient_weight.add(delta_gradient_weight);

                gradient_weights.set(i, new_gradient_weight);
            }

            /*Updating the network weights and biases base on the average gradient of the mini batches
            (with the learning rate)*/
            // Update the weights
            for(int i=0; i<this.weights.size(); i++){
                INDArray current_weight = this.weights.get(i);
                INDArray gradient_weight_sum = gradient_weights.get(i);

                INDArray average_weight_gradient = gradient_weight_sum.muli(learning_rate/mini_batches.size());
                INDArray new_weight = current_weight.sub(average_weight_gradient);

                this.weights.set(i, new_weight);
            }

            // Update the biases
            for(int i=0; i<this.biases.size(); i++){
                INDArray current_bias = this.biases.get(i);
                INDArray gradient_bias_sum = gradient_biases.get(i);

                INDArray average_bias_gradient = gradient_bias_sum.muli(learning_rate/mini_batches.size());
                INDArray new_bias = current_bias.sub(average_bias_gradient);

                this.biases.set(i, new_bias);
            }
        }
    }


    // This method is to evaluate the accuracy of the network with the test datas
    public int evaluate(List<List<INDArray>> test_datas){
        int correct_predictions = 0;
        List<List<Integer>> test_results = new ArrayList<>();

        for(List<INDArray> test_data:test_datas){
            INDArray raw_input = test_data.get(0);
            INDArray desired_output = test_data.get(1);

            INDArray finale_output_layer = this.feedforward(raw_input);
            Integer class_index = Nd4j.argMax(finale_output_layer).getInt(0);
            Integer class_result = Nd4j.argMax(desired_output).getInt(0);

            List<Integer> test_result = new ArrayList<>();
            test_result.add(class_index);
            test_result.add(class_result);

            test_results.add(test_result);
        }

        for(List<Integer> result:test_results){
            if(result.get(0) == result.get(1)) correct_predictions++;
        }

        return correct_predictions;
    }


    public void saveNetwork(String fileLocation) {
        try (ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(fileLocation))) {
            System.out.println("Saving current Network...");
            output.writeObject(this);
            System.out.println("Network saved.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static Network1 loadNetwork(String fileLocation) {
        Network1 net = null;
        try (ObjectInputStream input = new ObjectInputStream(new FileInputStream(fileLocation))) {
            System.out.println("Loading Network...");
            net = (Network1) input.readObject();
            System.out.println("Network loaded");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return net;
    }
}