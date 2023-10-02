package neural_network_project.ActivationFuncs;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;
import java.io.Serializable;

import java.util.Arrays;
import neural_network_project.Layers.Layer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.api.ops.impl.transforms.strict.TanhDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh;
//import org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative;

public class TanhA extends Layer implements Serializable{
    INDArray input;
    // Apply an activation func to a Vector, element-wise.
    @Override
    public INDArray forward(INDArray input){
        this.input = input;
        return Nd4j.getExecutioner().exec(new Tanh(input.dup()));
    }


    // comput the input gradient (cost derivative respect to input)
    @Override
    public INDArray backward(INDArray output_gradient){
        INDArray tanhDerivative = Nd4j.getExecutioner().exec(new TanhDerivative(this.input.dup()));
        return output_gradient.muli(tanhDerivative);
    }

    // Tell if this layer is trainable or not
    public boolean is_trainable(){
        return false;
    }


    @Override
    public String get_info(){
        return "Tanh Activation";
    }
}