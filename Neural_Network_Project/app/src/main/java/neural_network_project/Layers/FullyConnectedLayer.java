package neural_network_project.Layers;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;
import java.io.Serializable;

import javax.sound.midi.SysexMessage;

import java.util.Arrays;
import neural_network_project.Layers.Layer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;


public class FullyConnectedLayer extends Layer implements Serializable{
    int n_In;
    int n_Out;
    INDArray biases;
    INDArray biases_gradient;
    INDArray weights;
    INDArray weights_gradient;
    INDArray input;
    List<INDArray> parameters = new ArrayList<>();
    
    // Initalizing the Fully Connected Layer
    public FullyConnectedLayer(int n_In, int n_Out){
        this.n_In = n_In;
        this.n_Out = n_Out;

        // Initializing the parameters of the layer
        this.biases = Nd4j.randn(n_Out, 1);
        this.weights = Nd4j.randn(n_Out, n_In);

        // Assign the parameters list
        this.parameters.add(this.weights);
        this.parameters.add(this.biases);
    }


    // Method to forward pass the input through the layer
    @Override
    public INDArray forward(INDArray a){
        this.input = a;
        // Here we pass the input and comput the output as usual.
        a = this.weights.mmul(a);
        a = a.add(this.biases);

        return a;
    }


    /*This is the backward pass, given the output gradient (derivative of 
    cost func with respect to output), calculate the derivative with respect to weights and biases 
    (parameters gradients) and then return the input gradient*/
    @Override
    public INDArray backward(INDArray output_gradient){
        // Comput weights gradient given output gradient
        this.weights_gradient = output_gradient.mmul(this.input.transpose());

        // We don't compute the biases gradient since its equal to output_gradient
        this.biases_gradient = output_gradient;

        // Here we comput the input_gradient (deriavtive of cost func respect to layer input)
        INDArray input_gradient = this.weights.transpose().mmul(output_gradient);
        
        return input_gradient;
    }


    @Override
    public void update_mini_batch(INDArray weights_gradient, INDArray biases_gradient, float learning_rate, int mini_batch_size){
        // Calculate the average gradient of the mini batch and update the parameters with learning rate
        INDArray average_weights_gradient = weights_gradient.muli(learning_rate/mini_batch_size);
        this.weights.subi(average_weights_gradient);

        INDArray average_biases_gradient = biases_gradient.muli(learning_rate/mini_batch_size);
        this.biases.subi(average_biases_gradient);
    }


    // This getter method is to get the weights gradients of the current layer
    @Override
    public INDArray get_weights_gradients(){
        return this.weights_gradient;
    }

    // This getter method is to get the biases gradients of the current layer
    @Override
    public INDArray get_biases_gradients(){
        return this.biases_gradient;
    }

    // This method returns the current layer's weights
    @Override
    public INDArray get_weights(){
        return this.weights;
    }

    // This method returns the current layer's biases
    @Override
    public INDArray get_biases(){
        return this.biases;
    }

    // Tell if this layer is trainable or not
    @Override
    public boolean is_trainable(){
        return true;
    }


    @Override
    public String get_info(){
        return String.format("Fully Connected Layer (%d,%d)", this.n_In, this.n_Out);
    }
}