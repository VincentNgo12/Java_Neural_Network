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


public class HardTanh extends Layer implements Serializable{
    INDArray input;
    // Apply an activation func to a Vector, element-wise.
    @Override
    public INDArray forward(INDArray input){
        this.input = input;
        return Transforms.hardTanh(input);
    }


    // comput the input gradient (cost derivative respect to input)
    @Override
    public INDArray backward(INDArray output_gradient){
        return output_gradient.muli(Transforms.hardTanhDerivative(this.input));
    }

    // Tell if this layer is trainable or not
    public boolean is_trainable(){
        return false;
    }


    @Override
    public String get_info(){
        return "Hard Tanh Activation";
    }
}