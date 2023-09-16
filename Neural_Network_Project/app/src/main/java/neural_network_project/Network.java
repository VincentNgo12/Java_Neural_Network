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
    Layer[] layers;
    int num_layers;

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


    public List<List<INDArray>> backpropagation(INDArray input, INDArray desiredOutput){
        // Create lists to store the gradient vectors and matricies of biases and weights
        List<INDArray> gradient_biases = new ArrayList<>();
        List<INDArray> gradient_weights = new ArrayList<>();

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

        // Backpropagate through the layers (going backward)
        for(int i=this.num_layers-1; i>=0; i--){
            Layer layer = this.layers[i];
            // If the layer is not trainable (activation layers)
            if(!layer.is_trainable()){
                grad = layer.backward(grad);
            }

            grad = layer.backward(grad);
            gradient_weights.add(layer.get_weights_gradients());
            biases_gradient.add(layer.get_biases_gradients());
        }

        // reverse the gradient list because we have been adding it backwards
        Collections.reverse(gradient_weights);
        Collections.reverse(gradient_biases);

        // We forged the two gradients to a list and return it so other methods can use it.
        List<List<INDArray>> gradients = new ArrayList<>();
        gradients.add(gradient_weights);
        gradients.add(gradient_biases);
        return gradients;
    }


    public void update_mini_batch(List<List<INDArray>> mini_batches, float learning_rate){
        
    }


    public void train(List<List<INDArray>> training_datas, int epochs, float learning_rate, List<List<INDArray>> test_datas){
        for(int e=0; e<epochs; e++){
            for(List<INDArray> training_data: training_datas){
                INDArray output = training_data.get(0);
                INDArray y = training_data.get(1);

                for(Layer layer: this.layers){
                    output = layer.forward(output);
                }

                INDArray grad = cost_derivative(output,y);
                for(int i=1; i<this.layers.length; i++){
                    Layer backLayer = this.layers[this.layers.length-1];
                    grad = backLayer.backward(grad, learning_rate);
                }
            }
            // Print out the evaluated accuracy on current epoch
            System.out.println(String.format("Epoch %d: %d / %d", e, this.evaluate(test_datas), 10000));
        }
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
}