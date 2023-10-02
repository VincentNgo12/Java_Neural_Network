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
import org.nd4j.linalg.convolution.Convolution;


public class FullyConnectedLayer extends Layer implements Serializable{
    int input_depth;
    int input_height;
    int input_width;
    int depth;
    int output_height;
    int output_width;
    int kernel_size;
    INDArray input;
    INDArray output;
    INDArray biases;
    INDArray biases_gradient;
    INDArray kernels;
    INDArray kernels_gradient;

    
    // Initalizing the Fully Connected Layer
    public FullyConnectedLayer(int input_height, int input_width, int input_depth, int kernel_size, int depth){
        this.input_height = input_height;
        this.input_width = input_width;
        this.input_depth = input_depth;
        this.depth = depth;


        // Initializing the parameters of the layer
        this.biases = Nd4j.randn(depth, input_height - kernel_size + 1, input_width - kernel_size + 1);
        this.kernels = Nd4j.randn(depth, input_depth, kernel_size, kernel_size);

    }


    // Method to forward pass the input through the layer
    @Override
    public INDArray forward(INDArray input){
        this.input = input;
        this.output = self.biases.dup();

        for(int i=0; i<self.depth; i++){
            for(int j=0; j<self.input_depth; j++){
                INDArray convoled_image = Convolution.convolve(self.input.slice(j), self.kernels.slice(i,j), Convolution.Type.VALID);
                self.output.slice(i).sumi(convoled_image);
            }
        }

        return self.output;
    }


    /*This is the backward pass, given the output gradient (derivative of 
    cost func with respect to output), calculate the derivative with respect to weights and biases 
    (parameters gradients) and then return the input gradient*/
    @Override
    public INDArray backward(INDArray output_gradient){
        this.kernels_gradient = Nd4j.zerosLike(this.kernels);
        this.biases_gradient = output_gradient;

        INDArray input_gradient = Nd4j.zeros(this.input_depth, this.input_height, this.input_width);

        for(int i=0 ; i<this.depth; i++){
            for(int j=0; j<this.input_depth; j++){
                INDArray k_gradient = Convolution.convolve(self.input.slice(j), output_gradient.slice(i), Convolution.Type.VALID);
                this.kernels_gradient.putScalar(i, j, k_gradient);

                INDArray input_gradient_slice = input_gradient.slice(j);
                input_gradient_slice.addi(Convolution.convolve(output_gradient.slice(i), this.kernels.slice(i,j), Convolution.Type.FULL));
            }
        }

        return input_gradient;
    }


    @Override
    public void update_mini_batch(INDArray weights_gradient, INDArray biases_gradient, float learning_rate, int mini_batch_size){
        // Calculate the average gradient of the mini batch and update the parameters with learning rate
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
}