package neural_network_project.ActivationFuncs;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;
import java.io.Serializable;

import java.util.Arrays;
import neural_network_project.Layers.Layer;
import neural_network_project.Helper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;


public class Tanh extends Layer implements Serializable{
    INDArray input;
    // Apply an activation func to a Vector, element-wise.
    @Override
    public INDArray forward(INDArray input){
        this.input = input.dup();
        // Apply the custom tanh function to each element of the array
        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < input.columns(); j++) {
                double value = input.getDouble(i, j);
                double newValue = tanh(value);
                input.putScalar(i, j, newValue);
            }
        }
        return input;
    }


    // comput the input gradient (cost derivative respect to input)
    @Override
    public INDArray backward(INDArray output_gradient){
        // apply the tanh derivative to the input indarray
        for (int i = 0; i < this.input.rows(); i++) {
            for (int j = 0; j < this.input.columns(); j++) {
                double value = this.input.getDouble(i, j);
                double newValue = tanhDerivative(value);
                this.input.putScalar(i, j, newValue);
            }
        }
        return output_gradient.muli(this.input);
    }

    // Tell if this layer is trainable or not
    public boolean is_trainable(){
        return false;
    }


    // The tanh function
    public static double tanh(double x) {
        return Math.tanh(x);
    }

    //The tanh derivative function
    public static double tanhDerivative(double x) {
        return 1 - Math.pow(Math.tanh(x), 2);
    }
}