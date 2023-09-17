package neural_network_project.ActivationFuncs;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;

import java.util.Arrays;
import neural_network_project.Layers.Layer;
import neural_network_project.Helper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh;


public class TanhA extends Layer{
    INDArray input;
    // Apply an activation func to a Vector, element-wise.
    @Override
    public INDArray forward(INDArray input){
        this.input = input;
        Nd4j.getExecutioner().execAndReturn(new Tanh(input));
        return input;
    }


    // comput the input gradient (cost derivative respect to input)
    @Override
    public INDArray backward(INDArray output_gradient){
        INDArray epsilon = Nd4j.valueArrayOf(this.input.shape(), 1e-6f);
        Nd4j.getExecutioner().execAndReturn(new TanhDerivative(output_gradient, output_gradient, output_gradient));
        return output_gradient;
    }

    // Tell if this layer is trainable or not
    public boolean is_trainable(){
        return false;
    }
}