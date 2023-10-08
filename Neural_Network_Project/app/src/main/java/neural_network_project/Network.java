package neural_network_project;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;

import javax.sound.midi.SysexMessage;

import java.util.Arrays;
import neural_network_project.Layers.Layer;
import neural_network_project.Layers.FullyConnectedLayer;
import neural_network_project.ActivationFuncs.Sigmoid;
import neural_network_project.ActivationFuncs.Tanh;
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


public class Network implements Serializable{
    private static final long serialVersionUID = 1L;
    Layer[] layers;
    int num_layers;
    List<INDArray> biases_gradient;
    List<INDArray> weights_gradient;

    // Initialize the object
    public Network(Layer... layers){
        this.layers = layers;
        this.num_layers = layers.length;
    }


    // This methods feed forward throught the layers of the network
    public INDArray feedforward(INDArray input){
        for(Layer layer : layers){
            input = layer.forward(input);
        }
        return input;
    }


    public void stochastic_gradient_descent(List<List<INDArray>> training_datas, int epochs, int mini_batch_size, float learning_rate, float lambda, List<List<INDArray>> test_datas){
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
                this.update_mini_batch(mini_batch, learning_rate, lambda, mini_batch_size, n);
            }

            // Print out the evaluated accuracy on current epoch
            System.out.println(String.format("Epoch %d: %d / %d", i, this.evaluate(test_datas), n_test));
        }   
    }


    public void backpropagation(INDArray input, INDArray desiredOutput){
        // /*Initialize them with zeros, gradient vectors and matrices have to be the same size of the 
        // current biases and weights of the network*/ 
        // for(int i=0; i<this.num_layers; i++){
        //     Layer layer = this.layers[i];
        //     if(layer.is_trainable()){
        //         gradient_weights.add(Nd4j.zerosLike(layer.get_weights()));
        //         gradient_biases.add(Nd4j.zerosLike(layer.get_biases()));
        //     }
        // }


        // Start computing the gradients
        INDArray output = this.feedforward(input);

        // First, we need the gradient of the very last cost layer
        INDArray grad = cost_derivative(output,desiredOutput);
        // Keep track of the index of the network's list of gradients
        int gradientList_index = this.biases_gradient.size() - 1;


        // Backpropagate through the layers (going backward)
        for(int i=this.num_layers-1; i>=0; i--){
            Layer layer = this.layers[i];
            // If the layer is not trainable (activation layers)
            if(!layer.is_trainable()){
                grad = layer.backward(grad);
                continue;
            }

            grad = layer.backward(grad);
            // add up the computed gradient to the total gradients
            this.weights_gradient.get(gradientList_index).addi(layer.get_weights_gradients());
            this.biases_gradient.get(gradientList_index).addi(layer.get_biases_gradients());
            //Reduce gradientList_index after each trainable layer
            gradientList_index--;
        }

        return;
    }


    public void update_mini_batch(List<List<INDArray>> mini_batches, float learning_rate, float lambda, int mini_batch_size, int n){
        // reset the network's gradients lists
        this.biases_gradient = new ArrayList<>();
        this.weights_gradient = new ArrayList<>();

        /*Initialize them with zeros, gradient vectors and matrices have to be the same size of the 
        current biases and weights of the network*/ 
        for(int i=0; i<this.num_layers; i++){
            Layer layer = this.layers[i];
            if(layer.is_trainable()){
                this.weights_gradient.add(Nd4j.zerosLike(layer.get_weights()));
                this.biases_gradient.add(Nd4j.zerosLike(layer.get_biases()));
            }
        }

        // Iterate throught the mini batches
        for(List<INDArray> mini_batch : mini_batches){
            // Get the raw input and the correct labeled output
            INDArray input = mini_batch.get(0);
            INDArray desiredOutput = mini_batch.get(1);

            // Calling backpropagation method and let it compute the summud up gradients list
            this.backpropagation(input, desiredOutput);
        }

        int gradientList_index=0;
        // Iterate through the network's layers to apply the computed gradients
        for(Layer layer : this.layers){
            if(!layer.is_trainable()){
                continue;
            }

            // If the layer is trainable, get according gradient
            INDArray bias_gradient = this.biases_gradient.get(gradientList_index);
            INDArray weight_gradient = this.weights_gradient.get(gradientList_index);

            // Use the update_mini_batch method of each trainable layer
            layer.update_mini_batch(weight_gradient, bias_gradient, learning_rate, lambda, mini_batch_size, n);
            // Update the gradient list index
            gradientList_index++;
        }

        return;
    }


    // Comput the mean-square cost derivative
    public INDArray cost_derivative(INDArray output_activations, INDArray desiredOutput){
        return output_activations.sub(desiredOutput);
    }


    // Evaluate the accuracy of the network on test datas
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


    // This method is used to serialize an object of the Network class (usually a trained network)
    public void saveNetwork(String fileLocation) {
        try (ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(fileLocation))) {
            System.out.println("Saving current Network...");
            output.writeObject(this);
            System.out.println("Network saved.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    // This method is used to load a Network object from a serialized object
    public static Network loadNetwork(String fileLocation) {
        Network net = null;
        try (ObjectInputStream input = new ObjectInputStream(new FileInputStream(fileLocation))) {
            System.out.println("Loading Network...");
            net = (Network) input.readObject();
            System.out.println("Network loaded");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return net;
    }


    public void info(){
        for(int i = 0; i<this.layers.length; i++){
            Layer layer = this.layers[i];
            System.out.printf("Layer #%d: %s \n", i, layer.get_info());
        }
        return;
    }
}