package neural_network_project.Layers;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;

import javax.sound.midi.SysexMessage;

import java.util.Arrays;
import neural_network_project.Layers.Layer;
import neural_network_project.Helper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;


public class FullyConnectedLayer extends Layer{
    int n_In;
    int n_Out;
    INDArray biases;
    INDArray weights;
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
        // Activation on the output
        a = Transforms.sigmoid(a);

        return a;
    }


    /*This is the backward pass, given the output gradient (derivative of 
    cost func with respect to output), calculate the derivative with respect to weights and biases 
    (parameters gradients) and then update the layer's parameters*/
    @Override
    public INDArray backward(INDArray output_gradient, float learning_rate){
        // Comput weights gradient given output gradient
        INDArray weights_gradient = output_gradient.mmul(this.input.transpose());
        // updating the layer parameters
        this.weights.subi(weights_gradient.muli(learning_rate));
        // We don't compute the biases gradient since its equal to output_gradient
        this.biases.subi(output_gradient.mul(learning_rate));

        // Here we comput the input_gradient (deriavtive of cost func respect to layer input)
        INDArray input_gradient = this.weights.transpose().mmul(output_gradient);
        
        return input_gradient;
    }
}